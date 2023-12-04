from multiprocessing import Process,Pipe
import sounddevice as sd
import librosa
print(librosa.__version__)
import numpy as np

# import librosa.core as lc
# lc.fft_frequencies = librosa.fft_frequencies
# from config.config import SetConfig, EffectConfig, AudioHookConfig, EffectParamConfig
# import numpy  # Make sure NumPy is loaded before it is used in the callback
# assert numpy  # avoid "imported but unused" message (W0611)


def audio_listen(child_conn, input_device, channels, blocksize):

    def callback(indata, outdata, frames, time, status=None):
        if status:
            print(status)
        child_conn.send(indata)
        #print(libra.feature.chroma_stft(y=indata[:400], sr = 44100))
        # outdata[:] = indata


    with sd.InputStream(device=input_device,
                channels=channels, callback=callback, samplerate=44100, blocksize=blocksize):
        while True:
            continue

class AudioHandler(object):
    AudioHooks = ["Treble", "Mid", "Bass", "Turbulence", "RMS"]

    def __init__(self, input_device, channels, blocksize = 2048):
        self.input_device = input_device
        self.channels = channels
        self.started = False
        self.blocksize = blocksize
        self.audio = np.zeros((channels, blocksize))
        self.p = None
        self.parent_conn = None
        self.child_conn = None
        self.audio_hooks_to_compute = []
        self.audio_hook_values = {}
        self.fft_freqs = self.extract_fft_freqs()
        self.num_fft_freqs = self.fft_freqs.shape[0]
        self.normalization = {}
        self.norm_gamma = 0.01
        self.hook_smoothing = 0.7 # lower is less smoothing (kind of like release time)

        self.fft = np.zeros((channels, self.num_fft_freqs))
        self.prev_fft = np.zeros((channels, self.num_fft_freqs))

        # for effect_id, effect in set_config.effects.items():
        #     for param_name, param in effect.params.items():
        #         for hook_name, hook in param.audio_hooks.items():
        #             if hook_name not in self.audioHooksToCompute:
        #                 self.audioHooksToCompute.append(hook_name)
        for hook_name in self.AudioHooks:
            self.audio_hook_values[hook_name] = 0
            self.normalization[hook_name] = 0.00000000001
    
    def start(self):
        assert not self.started, "AudioHandler already started"
        self.started = True

        self.parent_conn, self.child_conn = Pipe()

        self.p = Process(target=audio_listen, args=(self.child_conn, self.input_device, self.channels, self.blocksize))
        self.p.start()

    def stop(self):
        self.child_conn.close()
        self.p.terminate()
    
    def listen(self):
        assert self.started, "must call start() before you can listen()"
        indata = self.parent_conn.recv() # shape (self.blocksize, ch)
        self.audio = np.transpose(indata) # shape (ch, self.blocksize)
        self.extract_fft()
        # self.extract_hooks()

    def extract_rms(self):
        rms = librosa.feature.rms(y=self.audio, hop_length=self.blocksize, center=False) # (ch, 1, num_windows = 1)
        return np.average(rms) # ()

    def extract_centroid(self):
        centroid = librosa.feature.spectral_centroid(y=self.audio, n_fft=self.blocksize, hop_length=self.blocksize,sr=44100, center=False) # (ch, 1, num_windows = 1)
        return np.sum(centroid, axis=2) # (ch, 1)

    def extract_fft(self):
        self.prev_fft = self.fft
        stft =  librosa.stft(y=self.audio, n_fft=self.blocksize, hop_length=self.blocksize, center=False) # (ch, num_freqs, num_windows = 1)
        self.fft = np.sum(stft, axis=2) # (ch, num_freqs)
        return self.fft

    def extract_fft_freqs(self):
        return librosa.fft_frequencies(sr=44100, n_fft=self.blocksize) # (num_freqs, )
    
    def get_hook_value(self, hook_name):
        assert hook_name in self.audio_hook_values.keys(), "Hook name doesn't exist"
        return self.audio_hook_values[hook_name]
    
    def set_hook_value(self, hook_name, val):
        new_val = self.normalize(hook_name, val)
        old_val = self.get_hook_value(hook_name)
        if (new_val > old_val * self.hook_smoothing):
            self.audio_hook_values[hook_name] = new_val
        else:
            self.audio_hook_values[hook_name] = old_val * self.hook_smoothing
    
    def extract_hooks(self):
        def hook_treble_mid_bass():
            x = np.average(self.fft, axis=0) # (num_freqs)
            x = np.array_split(x, 3) # (3, num_freqs / 3)
            x = list(map(lambda a : abs(np.average(a)), x)) # (3,)
            # x = np.average(x, axis=1) # (2,)
            self.set_hook_value("Bass", x[0])
            self.set_hook_value("Mid", x[1])
            self.set_hook_value("Treble", x[2])

        def hook_turbulence():
            new = np.real(np.average(self.fft, axis=0))
            old = np.real(np.average(self.prev_fft, axis=0))
            self.set_hook_value("Turbulence", abs(np.average(np.subtract(new, old))))
        
        def hook_rms():
            self.set_hook_value("RMS", self.extract_rms())
        # def hook_width():
        #     print(self.fft[0])
        #     left = np.real(self.fft[0])
        #     right = np.real(self.fft[1])
        #     width = abs(np.average(np.subtract(left, right)))
        #     self.audio_hook_values["Width"] = self.normalize("Width", width)

        self.extract_fft()
        hook_treble_mid_bass()
        hook_turbulence()
        hook_rms()

        # hook_width()

    def normalize(self, var, val):
        self.normalization[var] = max(self.normalization[var], val)
        normalized = val / self.normalization[var]
        self.normalization[var] *= (1 - self.norm_gamma)
        return normalized
    