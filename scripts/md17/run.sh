for name in malonaldehyde # benzene_old malonaldehyde ethanol toluene salicylic aspirin uracil
do
    for learning_rate in 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5
    do
        for depth in 3 4 5 6 7 8
        do
            for hidden_features in 32 64 128 256 512 # 256
            do
                for weight_decay in 0.0 # 1e-4 1e-5 1e-6 1e-7
                do

        bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 23:59 -n 1\
        python run.py \
        --data $name \
        --learning_rate $learning_rate \
        --depth $depth \
        --hidden_features $hidden_features \
        --weight_decay $weight_decay \
        --n_tr 1000 \
        --n_epoch 20000 \
        --batch_size 32 \
        --out "__"$name$hidden_features
    done
done
done
done
done
