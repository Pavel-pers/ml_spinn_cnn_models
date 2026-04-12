import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def __init__(
            self,
            metadata,
            dataset_path,
            label_encoder,
            target_length=4,
            sr=22050,
            n_mels=128,
            augment=False,
            cache_dir=None,
            aug_double_masking=False,
            aug_time_mask_param=15,
            aug_freq_mask_param=25,
            global_mean: float | None = None,
            global_std: float | None = None,
    ):
        """
        Реализуйте датасет для UrbanSound8K.

        Что стоит продумать:
        - как хранить metadata и переиндексировать его
        - как загружать аудио и приводить его к фиксированной длине
        - как строить и нормализовать мел-спектрограммы
        - как организовать кэш, чтобы не пересчитывать спектрограммы каждый раз
        - какие аугментации допустимо применять только на train
        """

        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=512, n_mels=n_mels
        )

        self.target_length = target_length
        self.target_sr = sr
        self.n_mels = n_mels
        self.augment = augment
        self.aug_double_masking = aug_double_masking

        self.global_mean = global_mean
        self.global_std = global_std

        self.mel_data = []
        self.label_data = []

        self.time_aug = torchaudio.transforms.TimeMasking(aug_time_mask_param)
        self.freq_aug = torchaudio.transforms.FrequencyMasking(aug_freq_mask_param)

        for ind, row in metadata.iterrows():
            fold_ind = row['fold']
            mov_name = row['slice_file_name']
            mov_path = Path(dataset_path) / 'audio' / f'fold{fold_ind}' / mov_name

            processed_waveform = self.load_audio(mov_path,
                                                 Path(cache_dir) / f'{mov_name}.pt' if cache_dir is not None else None)
            self.mel_data.append(processed_waveform)
            self.label_data.append(label_encoder.transform([row['class']])[0])

    def __len__(self):
        return len(self.mel_data)

    def load_audio(self, file_path, cache_path: Path | None = None):
        """
        Загрузить аудио, привести к нужной частоте дискретизации
        и длине target_length
        """
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        waveform, origin_sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if origin_sr != self.target_sr:
            transformation = torchaudio.transforms.Resample(orig_freq=origin_sr, new_freq=self.target_sr)
            waveform = transformation(waveform)

        target_dim = self.target_length * self.target_sr
        if waveform.shape[-1] > target_dim:
            new_start = (waveform.shape[-1] - target_dim) // 2
            waveform = waveform[..., new_start:new_start + target_dim]
        if waveform.shape[-1] < target_dim:
            delta = target_dim - waveform.shape[-1]
            waveform = F.pad(waveform, (delta // 2, delta - delta // 2), mode="constant", value=0)

        processed_mel = self.extract_mel_spectrogram(waveform)
        if cache_path is not None:
            torch.save(processed_mel, cache_path)
        return processed_mel

    def extract_mel_spectrogram(self, y):
        """
        Построить мел-спектрограмму и перевести её в dB-шкалу
        """
        mel_y = self.mel_transform(y)
        dB_y = UrbanSoundDataset.amp_to_db(mel_y)
        return dB_y

    def normalize(self, y):
        if self.global_mean is not None and self.global_std is not None:
            y = (y - self.global_mean) / (self.global_std + 1e-6)
        else:
            y = (y - y.mean()) / (y.std() + 1e-6)
        return y

    def augment_audio(self, y):
        """
        Реализовать аугментации аудио
        Использовать только если self.augment=True
        """

        if not self.augment:
            return y

        noise_std = random.uniform(0.005, 0.02)
        noise = torch.randn_like(y) * noise_std * (y.max() - y.min())
        y = y + noise

        y = self.time_aug(y)
        y = self.freq_aug(y)
        if self.aug_double_masking:
            y = self.time_aug(y)
            y = self.freq_aug(y)

        return y

    def __getitem__(self, idx):
        """
        Вернуть пару (mel_spectrogram, label)
        """
        cur_data = self.normalize(self.mel_data[idx].clone())
        return self.augment_audio(cur_data), self.label_data[idx]

    def calc_mean_std(self):
        mel_np = torch.stack(self.mel_data)
        mean = mel_np.mean().item()
        std = mel_np.std().item()
        return mean, std
