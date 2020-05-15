# gpu4

nohup python main.py --gpu=3 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=vanilla --dec_self_attn=vanilla --attn_norm=softmax --machine_name=gpu4 --name=vanilla &> nohup/vanilla &

nohup python main.py --gpu=4 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense --dec_self_attn=dense --attn_norm=softmax --machine_name=gpu4 --name=dense_softmax &> nohup/dense_softmax &

nohup python main.py --gpu=5 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=random --dec_self_attn=random --attn_norm=softmax --machine_name=gpu4 --name=random_softmax &> nohup/random_softmax &

nohup python main.py --gpu=6 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense_random --dec_self_attn=dense_random --attn_norm=softmax --machine_name=gpu4 --name=dense_random_softmax &> nohup/dense_random_softmax &

nohup python main.py --gpu=7 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense --dec_self_attn=dense --attn_norm=tanh --machine_name=gpu4 --name=dense_tanh &> nohup/dense_tanh &

# gpu5

nohup python main.py --gpu=1 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=random --dec_self_attn=random --attn_norm=tanh --machine_name=gpu5 --name=random_tanh &> nohup/random_tanh &

nohup python main.py --gpu=2 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense_random --dec_self_attn=dense_random --attn_norm=tanh --machine_name=gpu5 --name=dense_random_tanh &> nohup/dense_random_tanh &

nohup python main.py --gpu=3 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense --dec_self_attn=dense --attn_norm=sigmoid --machine_name=gpu5 --name=dense_sigmoid &> nohup/dense_sigmoid &

nohup python main.py --gpu=4 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=random --dec_self_attn=random --attn_norm=sigmoid --machine_name=gpu5 --name=random_sigmoid &> nohup/random_sigmoid &

nohup python main.py --gpu=5 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense_random --dec_self_attn=dense_random --attn_norm=sigmoid --machine_name=gpu5 --name=dense_random_sigmoid &> nohup/dense_random_sigmoid &

nohup python main.py --gpu=6 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense --dec_self_attn=dense --attn_norm=softmax --sparsity_mode=topk --machine_name=gpu5 --name=dense_softmax_topk &> nohup/dense_softmax_topk &

nohup python main.py --gpu=7 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=random --dec_self_attn=random --attn_norm=softmax --sparsity_mode=topk --machine_name=gpu5 --name=random_softmax_topk &> nohup/random_softmax_topk &

# gpu7

nohup python main.py --gpu=0 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=dense_random --dec_self_attn=dense_random --attn_norm=softmax --sparsity_mode=topk --machine_name=gpu7 --name=dense_random_softmax_topk &> nohup/dense_random_softmax_topk &

nohup python main.py --gpu=1 --debug_mode=0 --p_dropout=0  --enc_self_attn=dense --dec_self_attn=dense --attn_norm=softmax --sparsity_mode=topk --machine_name=gpu7 --name=dense_softmax_topk_nodropout &> nohup/dense_softmax_topk_nodropout &

nohup python main.py --gpu=2 --debug_mode=0 --p_dropout=0  --enc_self_attn=random --dec_self_attn=random --attn_norm=softmax --sparsity_mode=topk --machine_name=gpu7 --name=random_softmax_topk_nodropout &> nohup/random_softmax_topk_nodropout &

nohup python main.py --gpu=4 --debug_mode=0 --p_dropout=0  --enc_self_attn=dense_random --dec_self_attn=dense_random --attn_norm=softmax --sparsity_mode=topk --machine_name=gpu7 --name=dense_random_softmax_topk_nodropout &> nohup/dense_random_softmax_topk_nodropout &

nohup python main.py --gpu=5 --debug_mode=0 --p_dropout=0.1  --enc_self_attn=vanilla --dec_self_attn=vanilla --attn_norm=softmax --sparsity_mode=topk --machine_name=gpu7 --name=vanilla_softmax_topk &> nohup/vanilla_softmax_topk &
