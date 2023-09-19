#%%
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
musicgen_model_version="large"

model = MusicGen.get_pretrained(musicgen_model_version, device=device)

audio_fp = "assets/boring_drums.wav"
prompt = "simple drum beat"

attributes, _ = model._prepare_tokens_and_attributes([prompt], None)

conditions = attributes
#null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
#conditions = conditions + null_conditions
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

with model.autocast:
    lm_output = model.lm.compute_predictions(
        codes=encoded_audio,
        conditions=[],
        condition_tensors=condition_tensors
    )

logits, logits_mask = lm_output.logits[0:1], lm_output.mask[0:1] 

N_CODES = logits.shape[1]-1

VOCAB_SIZE = logits.shape[-1]

# crop away N_CODES timsteps from end to avoid nans from interleave
logits = logits[:, :, :-N_CODES,:]
# offset logits by 1 by padding with ones
logits = torch.cat([torch.ones_like(logits[:, :, :1, :]), logits], dim=2)
temp=4.0
# apply softmax to get probabilities
probs = torch.softmax(logits / temp, dim=-1)
# now compute surprisal of tokens
log_probs = torch.log(probs)

# compute entropy of logit distribution
entropy = -(probs * log_probs).sum(dim=-1)

# one hot encode encoded_audio
encoded_audio_1hot = torch.nn.functional.one_hot(encoded_audio.long(), num_classes=VOCAB_SIZE).float()

# crop away N_CODES timsteps from end to same as logits
encoded_audio_1hot = encoded_audio_1hot[:, :, :logits.shape[2], :]
self_information = -(log_probs*encoded_audio_1hot).sum(dim=-1)

# print shapes
adjusted_surprise = -((log_probs * encoded_audio_1hot).sum(dim=-1)/entropy)

sns.set_theme()

gt_probs = probs * encoded_audio_1hot
gt_probs = gt_probs.sum(dim=-1)

# gaussian smoothing sigma
SIGMA= 1
def smooth(x, sigma=SIGMA):
    # numpy convert to float32 
    x = x.astype(np.float32)
    return gaussian_filter1d(x, sigma=sigma)

# sum over tokens
entropy = entropy.mean(dim=1, keepdim=True)
self_information = self_information.mean(dim=1, keepdim=True)
adjusted_surprise = adjusted_surprise.mean(dim=1, keepdim=True)
gt_probs = gt_probs.mean(dim=1, keepdim=True)

for token in [0]:
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
    plt.plot(smooth(entropy[0, token, :].detach().cpu().numpy()))
    plt.title("Entropy")
    plt.xticks([])

    plt.subplot(9,1,3)
    plt.plot(smooth(self_information[0, token, :].detach().cpu().numpy()))
    plt.title("Self Information")
    plt.xticks([])

    plt.subplot(9,1,4)
    plt.plot(smooth(adjusted_surprise[0, token, :].detach().cpu().numpy()))
    plt.title("Adjusted Surprise")
    plt.xticks([])

    plt.subplot(9,1,5)
    plt.plot(smooth(gt_probs[0, token, :].detach().cpu().numpy()))
    plt.title("Ground Truth Probability")
    plt.xticks([])

   
    # take first derivative of self info, entropy, adjusted surprise
    plt.subplot(9,1,6)
    plt.plot(smooth(np.gradient(entropy[0, token, :].detach().cpu().numpy())))
    plt.title("Entropy Derivative")
    plt.xticks([])

    plt.subplot(9,1,7)
    plt.plot(smooth(np.gradient(self_information[0, token, :].detach().cpu().numpy())))
    plt.title("Self Information Derivative")
    plt.xticks([])

    plt.subplot(9,1,8)
    plt.plot(smooth(np.gradient(adjusted_surprise[0, token, :].detach().cpu().numpy())))
    plt.title("Adjusted Surprise Derivative")
    plt.xticks([])

    plt.subplot(9,1,9)
    plt.plot(smooth(np.gradient(gt_probs[0, token, :].detach().cpu().numpy())))
    plt.title("Ground Truth Probability Derivative")
    plt.xticks([])
    
    # add supertitle
    plt.suptitle(f"Audio file: {audio_fp.split('/')[-1]}, Text prompt: {prompt},\nSmoothing Sigma: {SIGMA}, Temp: {temp}, Model: {musicgen_model_version}")
    plt.savefig(f"plots/{audio_fp.split('/')[-1]}_{prompt}_{SIGMA}_{temp}_{musicgen_model_version}.png", dpi=300)

    plt.show()
    plt.savefig
    plt.close()



display_audio(x, 32000)


# %%
