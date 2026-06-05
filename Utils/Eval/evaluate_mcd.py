import sys,os,torch,numpy,soundfile,pysptk
from scipy import spatial
from fastdtw import fastdtw

def sptk_extract(
    x: numpy.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
):
    """Extract SPTK-based mel-cepstrum.
    Args:
        x (ndarray): 1D waveform array.
        fs (int): Sampling rate
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
        is_padding (bool): Whether to pad the end of signal (default=False).
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = numpy.pad(x, (0, n_pad), "reflect")

    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1

    # get window function
    win = pysptk.sptk.hamming(n_fft)

    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calculate spectrogram
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return numpy.stack(mcep)

def _get_best_mcep_params(fs: int):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")

def main():

    if len(sys.argv)!=3:
        print("Usage: python",sys.argv[0],"ref-pt-dir hyp-pt-dir")
        exit(100)

    D=[]
    F=[]
    for f in os.listdir(sys.argv[1]):
        n,e=os.path.splitext(f)
        if e!=".wav": continue
        F.append(n)
        g=os.path.join(sys.argv[2],n+e)
        if not os.path.exists(g): raise ValueError("File not found: "+g)
        gen_x, gen_fs = soundfile.read(g, dtype="float64")
        gt_x, gt_fs = soundfile.read(os.path.join(sys.argv[1],f), dtype="float64")
        if gen_fs != gt_fs: raise ValueError("Sampling rate mismatch")
        fs=gen_fs
        #r=torch.load(os.path.join(sys.argv[1],f)).t().detach().numpy()
        #h=torch.load(g).detach().numpy()
        gen_mcep = sptk_extract(
                x=gen_x,
                fs=fs,
                n_fft=1024, #args.n_fft,
                n_shift=256, #args.n_shift,
                mcep_dim=None, #args.mcep_dim,
                mcep_alpha=None, #args.mcep_alpha,
        )
        gt_mcep = sptk_extract(
                x=gt_x,
                fs=fs,
                n_fft=1024,
                n_shift=256,
                mcep_dim=None, #args.mcep_dim,
                mcep_alpha=None, #args.mcep_alpha,
        )
        # DTW (below from espnet)
        _,path=fastdtw(gen_mcep,gt_mcep,dist=spatial.distance.euclidean)
        twf=numpy.array(path).T
        h_dtw=gen_mcep[twf[0]]
        r_dtw=gt_mcep[twf[1]]

        # MCD
        diff2sum=numpy.sum((h_dtw-r_dtw)**2,1)
        mcd=numpy.mean(10.0/numpy.log(10.0)*numpy.sqrt(2*diff2sum),0)
        #print(n,mcd)
        D.append(mcd)

    D=numpy.array(D)
    mean_mcd=numpy.mean(D)
    std_mcd=numpy.std(D)
    print(mean_mcd,"+/-",std_mcd)

    F, D=zip(*sorted(zip(F, D)))
    print("Top-3")
    print(F[0],D[0])
    print(F[1],D[1])
    print(F[2],D[2])
    print("Bottom-3")
    print(F[-1],D[-1])
    print(F[-2],D[-2])
    print(F[-3],D[-3])

    with open(sys.argv[2]+"/utt2mcd.log",mode='w') as w:
        for i in range(len(F)):
            w.write(F[i]+" "+str(float(D[i]))+"\n")

    with open(sys.argv[2]+"/evaluation_results.txt", "w") as f:
        f.write(f"#utterances: {len(F)}\n")
        f.write(f"Average: {mean_mcd:.4f} ± {std_mcd:.4f}")

    return f"{mean_mcd:.4f} ± {std_mcd:.4f}"

if __name__ == "__main__":
    results = main()
    print(results)
