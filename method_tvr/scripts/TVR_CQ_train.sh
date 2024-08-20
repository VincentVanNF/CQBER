#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash method/scripts/train.sh tvr all ANY_OTHER_PYTHON_ARGS
# use --eval_tasks_at_training ["VR", "SVMR", "VCMR"] --stop_task ["VR", "SVMR", "VCMR"] for
# use --lw_neg_q 0 --lw_neg_ctx 0 for training SVMR/SVMR only
# use --lw_st_ed 0 for training with VR only
export CUDA_VISIBLE_DEVICES=2
dset_name=$1  # see case below
ctx_mode=$2  # [video, sub, tef, video_sub, video_tef, sub_tef, video_sub_tef]
vid_feat_type=$3  # [resnet, i3d, resnet_i3d]
feature_root=features
data_root=data
results_root=method_tvr/results
vid_feat_size=2048
extra_args=()



if [[ ${ctx_mode} == *"sub"* ]] || [[ ${ctx_mode} == "sub" ]]; then
    if [[ ${dset_name} != "tvr" ]]; then
        echo "The use of subtitles is only supported in tvr."
        exit 1
    fi
fi

MODIFY_DESC_BERT_PATH=''
QUERY_IMAGE_PATCHES_PATH=''
TVR_CQ_VIDEO_FEAT_PATH=''
case ${dset_name} in
    tvr)
        train_path=$data_root"/TVR-CQ_train.jsonl"
        video_duration_idx_path=$data_root"/TVR-CQ_video2dur_idx.json"
        desc_bert_path=${feature_root}"${MODIFY_DESC_BERT_PATH}"
        query_image_patches_path=${feature_root}"${QUERY_IMAGE_PATCHES_PATH}"
        extra_args+=(--query_image_patches)
        extra_args+=(49)
        extra_args+=(--query_image_size)
        extra_args+=(768)

        if [[ ${vid_feat_type} == "i3d" ]]; then
            echo "Using I3D feature with shape 1024"
            vid_feat_path=${feature_root}tvr/tvr_i3d_rgb600_avg_cl-1.5.h5
            vid_feat_size=1024
        elif [[ ${vid_feat_type} == "resnet" ]]; then
            echo "Using ResNet feature with shape 2048"
            vid_feat_path=${feature_root}tvr/tvr_resnet152_rgb_max_cl-1.5.h5
            vid_feat_size=2048
        elif [[ ${vid_feat_type} == "resnet_i3d" ]]; then
            echo "Using concatenated ResNet and I3D feature with shape 2048+1024"
            vid_feat_path=${feature_root}"{TVR_CQ_VIDEO_FEAT_PATH}"
            vid_feat_size=3072
            extra_args+=(--no_norm_vfeat)  # since they are already normalized.
        fi
        eval_split_name=val
        nms_thd=-1
        extra_args+=(--eval_path)
        extra_args+=($data_root"/TVR_CQ_val.jsonl")
        clip_length=1.5
        # extra_args+=(--max_ctx_l)
        # extra_args+=(100)  # max_ctx_l = 100 for clip_length = 1.5, only ~109/21825 has more than 100.
        extra_args+=(--max_pred_l)
        extra_args+=(16)
        if [[ ${ctx_mode} == *"sub"* ]] || [[ ${ctx_mode} == "sub" ]]; then
            echo "Running with sub."
            desc_bert_path=${feature_root}"${MODIFY_DESC_BERT_PATH}"  # overwrite
            sub_bert_path=${feature_root}"${QUERY_IMAGE_PATCHES_PATH}"
            sub_feat_size=768
            extra_args+=(--sub_feat_size)
            extra_args+=(${sub_feat_size})
            extra_args+=(--sub_bert_path)
            extra_args+=(${sub_bert_path})
        fi
        ;;
    *)
        echo -n "Unknown argument"
        ;;
esac

echo "Start training with dataset [${dset_name}] in Context Mode [${ctx_mode}]"
echo "Extra args ${extra_args[@]}"

python method_tvr/train.py \
--dset_name=${dset_name} \
--eval_split_name=${eval_split_name} \
--nms_thd=${nms_thd} \
--results_root=${results_root} \
--train_path=${train_path} \
--desc_bert_path=${desc_bert_path} \
--query_image_patches_path=${query_image_patches_path} \
--video_duration_idx_path=${video_duration_idx_path} \
--vid_feat_path=${vid_feat_path} \
--clip_length=${clip_length} \
--vid_feat_size=${vid_feat_size} \
--ctx_mode=${ctx_mode} \
${extra_args[@]} \
${@:4}