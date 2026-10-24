import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Assuming assistant.py is in the same directory
from assistant import Mic, Waker, Speaker, record_utterance, transcribe, chat

# Mocking audio device information structure (if needed for __init__)
class MockDeviceInfo:
    """Mock class for info dictionary used in __init__."""
    default_samplerate = 16000
    
# --- Mock Setup ---

# Helper class to mock the audio stream setup
class MockAudioDevice:
    def __init__(self):
        self.sample_rate = 16000
    
    def open(self, rate, channels):
        return self # Simulate successful opening
    
    def flush(self):
        pass

# --- Unit Tests ---

class TestMic(unittest.TestCase):
    """Tests for the Mic class and related recording functionalities."""

    def setUp(self):
        # Patching the internal info dictionary access for cleaner setup
        self.mock_info = MagicMock()
        self.mock_info.default_samplerate = 16000
        
        # Initialize Mic by patching dependencies if necessary.
        # Since we don't see the full dependencies, we'll mock the class structure.
        self.mic = Mic()

    @patch('assistant.info', self.mock_info)
    @patch('assistant.MockAudioDevice', autospec=True)
    def test_mic_initialization(self, MockAudioDevice):
        """Tests Mic initialization and device opening."""
        # Assuming Mic.__init__ uses info and opens a device
        self.mock_audio_instance = MockAudioDevice.return_value
        self.mock_audio_instance.open.return_value = None
        
        # Re-initialize mic if necessary, or mock the relevant method
        # Due to complexity, we assume the constructor logic is tested or we mock its output.
        # For now, we'll just assert it was called with expected parameters if possible.
        
        # A more robust test would require knowing the exact internal calls, 
        # but we confirm mocking capability here.
        self.mic.__init__()
        MockAudioDevice.return_value.open.assert_called()


    @patch('assistant.info', self.mock_info)
    @patch('assistant.Mic')
    def test_read(self, MockMic):
        """Tests the read method, simulating captured audio."""
        # Mock the instance method being called
        mock_instance = MockMic.return_value
        mock_instance.read.return_value = np.array([0.1, 0.2]) # Mock audio data
        
        # Assuming Mic's read method reads audio chunks
        self.mic = Mic() # Re-initialize to use the patched class
        # Simulate using the mic object
        self.mic.read()
        mock_instance.read.assert_called()


    @patch('assistant.info', self.mock_info)
    @patch('builtins.open', new_callable=MagicMock)
    def test_save_wav(self, mock_open):
        """Tests saving recorded PCM data to a WAV file."""
        mock_pcm = np.array([0.1, 0.2])
        test_path = "/tmp/test_output.wav"
        self.mic = Mic() # Re-initialize
        
        # The actual implementation of save_wav is mocked here
        # We primarily check if the file is opened and written to.
        self.mic.save_wav(mock_pcm, path=test_path)
        mock_open.assert_called_with(test_path, 'wb')


class TestWaker(unittest.TestCase):
    """Tests for the Waker class and wake word detection."""
    
    @patch('assistant.Mic')
    def test_waker_wait(self, MockMic):
        """Tests the wait loop for wake word detection."""
        waker = Waker()
        # Mock the Mic object's read method to simulate streaming data
        mock_mic = MockMic.return_value
        mock_mic.read.side_effect = [
            np.zeros(100), # Non-wake word data
            np.array([0.5]) # Signal that the wake word was heard
        ]
        
        # When wait returns, it should return True (or break the loop)
        with patch.object(waker, 'wait', return_value=True):
            waker.wait(mock_mic)
            # Assertions would go here to check the correct number of reads or exit condition

class TestSpeaker(unittest.TestCase):
    """Tests for the Speaker class and audio playback."""

    @patch('assistant.info', new_callable=MagicMock)
    @patch('assistant.audio_player', MagicMock()) # Assuming this is the underlying audio playback module
    def test_say_text(self, mock_audio_player):
        """Tests the say method for synthesizing and playing text."""
        speaker = Speaker()
        text_to_speak = "Hello world."
        speaker.say(text_to_speak)
        
        # We expect the underlying synthesis/play method to be called
        mock_audio_player.say.assert_called_once_with(text_to_speak)

    @patch('assistant.info', new_callable=MagicMock)
    @patch('assistant.audio_player', MagicMock())
    def test_beep(self, mock_audio_player):
        """Tests the beep functionality."""
        speaker = Speaker()
        speaker.beep(freq=1000, dur=0.1)
        
        # Assuming beep calls a low-level audio generation function
        mock_audio_player.beep.assert_called_once()

class TestCoreFunctions(unittest.TestCase):
    """Tests for core functionality like transcription and chat."""

    @patch('assistant.info', new_callable=MagicMock)
    @patch('assistant.Mic')
    @patch('assistant.transcribe', return_value="mocked transcript")
    def test_record_utterance(self, mock_transcribe, MockMic):
        """Tests the full cycle of recording and transcribing."""
        mic_instance = Mic() # Re-initialize for context
        
        # We expect the recording to happen and then transcribe the result
        record_utterance(mic_instance)
        mock_transcribe.assert_called_once()


    @patch('assistant.info', new_callable=MagicMock)
    @patch('assistant.transcribe')
    @patch('assistant.Mic')
    def test_transcribe(self, MockMic, mock_transcribe):
        """Tests the transcription process with dummy PCM data."""
        mock_pcm = np.array([0.1, 0.2])
        
        transcribe(mock_pcm)
        mock_transcribe.assert_called_once_with(mock_pcm)


    @patch('assistant.info', new_callable=MagicMock)
    @patch('assistant.Speaker')
    @patch('assistant.Waker')
    @patch('assistant.Mic')
    @patch('assistant.record_utterance')
    def test_chat_interaction_flow(self, mock_record, MockMic, MockWaker, MockSpeaker):
        """Tests the complete conversation flow (Waker -> Record -> Transcribe -> Speak)."""
        history = []
        
        # We mock the entire flow to ensure all components are hit
        chat(history, speaker=MockSpeaker())
        
        # Assertions would verify the flow sequence
        mock_record.assert_called_once()
        mock_speaker.say.assert_called_once()


if __name__ == '__main__':
    unittest.main()
