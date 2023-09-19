#%%
"""
Laura checking out other ways of plotting surprise
(mainly trying to figure out how musicgen works)
"""
from audiocraft.models import MusicGen
import torchaudio
import math
import torch
from audiocraft.utils.notebook import display_audio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

device = "cuda"
musicgen_model_version="medium"

model = MusicGen.get_pretrained(musicgen_model_version, device=device)

audio_fp = "assets/boring_drums.wav"
prompt = "simple drum beat"

attributes, _ = model._prepare_tokens_and_attributes([prompt], None)

conditions = attributes
tokenized = model.lm.condition_provider.tokenize(conditions)
cfg_conditions = model.lm.condition_provider(tokenized)

condition_tensors = cfg_conditions

# load audio
wav, sr = torchaudio.load(audio_fp)
wav = torchaudio.functional.resample(wav, sr, 32000) #32k is the model sr
wav = wav.mean(dim=0, keepdim=True).unsqueeze(0)
wav = wav.cuda()
x=wav

display_audio(x, 32000)

# encode audio with compression model
with torch.no_grad():
    gen_audio = model.compression_model.encode(wav)
codes, scale = gen_audio
encoded_audio = codes

# %%
with model.autocast:
    logits = model.lm(encoded_audio, conditions=[], condition_tensors=condition_tensors)

# %%
logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T] with K the number of 
                                     # codebooks and T the time steps

print("Logit shapes are: B, K (codebooks), card, T = ", logits.shape)

logits = logits[0, 0, ...] # remove batch and codebook dimension

# %%
temp=1.0

probs = torch.softmax(logits / temp, dim=0) # softmax over card

# grab the probabilities according to the encoded_audio 
# (i.e. the codebook indices)
probs = probs[encoded_audio[0, 0, ...], torch.arange(probs.shape[1])]

# now compute surprisal of tokens
info_content = - torch.log(probs + 1e-6)

# compute entropy of logit distribution
entropy = (probs * info_content)

# print shapes
adjusted_surprise = (info_content/(entropy + 1e-6))

# check that there are no nans in info_content, entropy, adjusted_surprise
assert not torch.isnan(info_content).any()
assert not torch.isnan(entropy).any()
assert not torch.isnan(adjusted_surprise).any()

sns.set_theme()

gt_probs = probs

plt.figure(figsize=(20, 20))

# reset rcParams
plt.rcParams.update(plt.rcParamsDefault)

# hide background grid
plt.rcParams["axes.grid"] = False

# do not show x axis ticks

plt.subplot(9,1,1)
plt.plot(x[0,0,:].detach().cpu().numpy())
plt.xticks([])
# put title to the left rotated by 90 degrees
# place it on the left side of the plot
plt.title("Waveform")

plt.subplot(9,1,2)
plt.plot(entropy.detach().cpu().numpy())
plt.title("Entropy")
plt.xticks([])

plt.subplot(9,1,3)
plt.plot(info_content.detach().cpu().numpy())
plt.title("Self Information")
plt.xticks([])

plt.subplot(9,1,4)
plt.plot(adjusted_surprise.detach().cpu().numpy())
plt.title("Adjusted Surprise")
plt.xticks([])

plt.subplot(9,1,5)
plt.plot(gt_probs.detach().cpu().numpy())
plt.title("Ground Truth Probability")
plt.xticks([])


# take first derivative of self info, entropy, adjusted surprise
plt.subplot(9,1,6)
plt.plot(np.gradient(entropy.detach().cpu().numpy()))
plt.title("Entropy Derivative")
plt.xticks([])

plt.subplot(9,1,7)
plt.plot(np.gradient(info_content.detach().cpu().numpy()))
plt.title("Self Information Derivative")
plt.xticks([])

plt.subplot(9,1,8)
plt.plot(np.gradient(adjusted_surprise.detach().cpu().numpy()))
plt.title("Adjusted Surprise Derivative")
plt.xticks([])

plt.subplot(9,1,9)
plt.plot(np.gradient(gt_probs.detach().cpu().numpy()))
plt.title("Ground Truth Probability Derivative")
plt.xticks([])

# add supertitle
plt.suptitle(f"Audio file: {audio_fp.split('/')[-1]}, Text prompt: {prompt},\n, Temp: {temp}, Model: {musicgen_model_version}")
plt.savefig(f"plots/laura_{audio_fp.split('/')[-1]}_{prompt}_{temp}_{musicgen_model_version}.png", dpi=300)

plt.show()
plt.savefig
plt.close()

display_audio(x, 32000)
