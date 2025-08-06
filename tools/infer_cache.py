# tools/infer_cache.py --------------------------------------------------
import os, argparse, numpy as np, torch, tqdm, config
from data.data_utils import load_and_preprocess_data
from models.unet3d   import UNet3D

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "val"], required=True)
parser.add_argument("--ckpt",  default=config.NET1_BEST)
parser.add_argument("--gpu",   default=config.GPU_NUMBER)
args = parser.parse_args()

# ---- Pfade & Daten ----------------------------------------------------
folders = config.train_data if args.split == "train" else config.val_data
outfile = config.CACHE_TRAIN if args.split == "train" else config.CACHE_VAL
os.makedirs(config.CACHE_DIR, exist_ok=True)

data = load_and_preprocess_data(
    folder_names = folders,
    base_path    = "datasets",
    fourier_axes = config.fourier_transform_axes,
    normalize    = True,
)                                             # (X,Y,Z,F,T) complex

# ---- Modell -----------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
net = UNet3D(config.in_channels, config.out_channels,
             config.features_3d).cuda()
net.load_state_dict(torch.load(args.ckpt)["model_state"])
net.eval()

# ---- Vollinferenz -----------------------------------------------------
out = np.empty_like(data, dtype=np.complex64)      # gleicher Shape
with torch.no_grad():
    for x in tqdm.trange(data.shape[0], desc=f"{args.split}: x"):
        for y in range(data.shape[1]):
            spec   = data[x, y]                                   # (Z,F,T) complex
            inp_np = np.stack([spec.real, spec.imag], 0)[None]    # (1,2,Z,F,T)
            pred   = net(torch.from_numpy(inp_np).cuda()).cpu().numpy()[0]
            den    = pred[0] + 1j*pred[1]                         # komplex
            out[x, y] = den

np.save(outfile, out.astype(np.complex64))
print(f"✓ Cache {args.split} gespeichert → {outfile}")



