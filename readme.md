to run PML-CD:

```bash
$ python PML_CD_train.py --dataset voc2007 --noise-rate 0.8
```

to train Calibrator

```bash
$ for noise_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python train.py --dataset voc2007 --noise-rate $noise_rate --weighting identity --export-ckpts voc2007-$noise_rate.export-ckpts;
  done
$ python calibrator_train.py "voc2007-*.ckpts" weighter_new.ckpt
```
