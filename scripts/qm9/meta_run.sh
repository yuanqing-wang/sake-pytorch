for lr in 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5
do
    for depth in 3 4 5 6 7 8 9
    do
        for width in 32 64 128 256
        do
            bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[ptile=1]" -W 23:59 -o %J.stdout \
                python run.py \
                --lr $lr \
                --depth $depth \
                --width $width
       done
    done
done
