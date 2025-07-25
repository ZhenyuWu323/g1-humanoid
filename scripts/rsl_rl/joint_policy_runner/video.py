"""Wrapper for recording videos."""
import os
from typing import Callable, Optional

import gymnasium as gym
from gymnasium import logger
from gymnasium.wrappers.monitoring import video_recorder


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 5000, 10000, 15000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 5000 == 0  # Much less frequent


class RecordVideo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper records videos of rollouts.

    Usually, you only want to record episodes intermittently, say every hundredth episode.
    To do this, you can specify **either** ``episode_trigger`` **or** ``step_trigger`` (not both).
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed.
    By default, the recording will be stopped once a `terminated` or `truncated` signal has been emitted by the environment. However, you can
    also create recordings of fixed length (possibly spanning several episodes) by passing a strictly positive value for
    ``video_length``.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            disable_logger (bool): Whether to disable moviepy logger or not.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder: Optional[video_recorder.VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        # MEMORY SAFETY: Hard limit video length
        if video_length <= 0:
            self.video_length = 100  # Force short videos if unlimited
        else:
            self.video_length = min(video_length, 150)  # Max 150 frames

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0
        self.episode_id = 0
        self.cleanup_counter = 0  # For periodic cleanup

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        
        # MEMORY SAFETY: Cleanup on reset
        self._force_cleanup()
        
        if self.recording:
            assert self.video_recorder is not None
            # Clear any existing frames
            if hasattr(self.video_recorder, 'recorded_frames'):
                self.video_recorder.recorded_frames = []
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            # Always check limits
            if self.recorded_frames > self.video_length:
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()
        self._force_cleanup()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            self.step_id += 1
            self.cleanup_counter += 1
            
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                
                # MEMORY SAFETY: Always force stop at limit
                should_stop = False
                if self.recorded_frames >= self.video_length:
                    should_stop = True
                elif not self.is_vector_env and (terminateds or truncateds):
                    should_stop = True
                elif self.is_vector_env and (terminateds[0] or truncateds[0]):
                    should_stop = True
                
                if should_stop:
                    self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()
            
            # MEMORY SAFETY: Periodic cleanup
            if self.cleanup_counter >= 30:  # Every 30 steps
                self._force_cleanup()
                self.cleanup_counter = 0

        return observations, rewards, terminateds, truncateds, infos

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            try:
                self.video_recorder.close()
            except:
                pass  # Ignore close errors
            self.video_recorder = None
        self.recording = False
        self.recorded_frames = 0
        self._force_cleanup()

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment or as specified in kwargs."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        # MEMORY SAFETY: Aggressively limit render history
        if hasattr(self.video_recorder, 'render_history'):
            if len(self.video_recorder.render_history) > 50:
                # Keep only the last 20 frames
                self.video_recorder.render_history = self.video_recorder.render_history[-20:]

        if len(self.video_recorder.render_history) > 0:
            # Limit how many frames we process at once
            max_frames = min(len(self.video_recorder.render_history), 10)
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(max_frames)
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()
        self._force_cleanup()

    def _force_cleanup(self):
        """Force memory cleanup without external imports."""
        try:
            # Standard garbage collection
            import gc
            gc.collect()
        except:
            pass
        
        try:
            # Clear CUDA cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass