for name in malonaldehyde # azobenzene naphthalene paracetamol malonaldehyde benzene_old malonaldehyde ethanol toluene salicylic aspirin uracil
do
    for learning_rate in 1e-3 # 1e-3 # 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5
    do
        for depth in 4 # 3 4 5 6 7 8 9 10 # 3 4 5 6 7 8  # 4 5 6 7 8
        do
            for hidden_features in 128 # 32 64 128 256 # 128 256 512  # 512 1024 # 128 # 32 64 128 256 512 # 256
            do
                for weight_decay in 1e-14 # 1e-16 1e-15 1e-14 1e-13 1e-12 # 1e-4 1e-5 1e-6 1e-7
                do
                    for n_coefficients in 128 # 16 32 64 128 256 512
                    do

        bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 11:59 -n 1\
        python run.py \
        --data $name \
        --learning_rate $learning_rate \
        --depth $depth \
        --hidden_features $hidden_features \
        --weight_decay $weight_decay \
        --n_tr 1000 \
        --n_epoch 10000 \
        --batch_size 32 \
        --n_coefficients $n_coefficients \
        --out "_"$weight_decay # $n_coefficients
    done
done
done
done
done
done