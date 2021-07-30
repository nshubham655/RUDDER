import copy
from typing import Dict, Union, List
from pathlib import Path

from typeguard import typechecked
from zsvision.zs_utils import memcache, concat_features

from utils import memory_summary
from base.base_dataset import BaseDataset


class MALTA(BaseDataset):

    @staticmethod
    @typechecked
    def dataset_paths() -> Dict[str, Union[str, List[str], Path, Dict]]:
        subset_paths = {}
        js_test_cap_idx_path = None
        challenge_splits = {"val"}
        splits = {"full-val","full-test"}
        splits.update(challenge_splits)
        for split_name in splits:

            train_list_path = "train_list_full.txt"
            if split_name == "full-val":
                test_list_path = "val_list_full.txt"
            else:
                test_list_path = "test_list_full.txt"

            subset_paths[split_name] = {"train": train_list_path, "val": test_list_path}
        feature_names = [
            "i3d.i3d.0",
            "imagenet.resnext101_32x48d.0",
            "imagenet.senet154.0",
            "r2p1d.r2p1d-ig65m.0",
            "scene.densenet161.0"
        ]
        custom_paths = {
            "audio": ["aggregated_audio_feats/Audio_TFT.pickle"],
        }
        custom_miech_paths = custom_paths.copy()
        custom_miech_paths.update({
        })
        text_feat_paths = {
            "marathi": "marathi.pickle",
#             "openai": "w2v_MSRVTT_openAIGPT.pickle",
#             "bertxl": "w2v_MSRVTT_transformer.pickle",
        }
        text_feat_paths = {key: Path("aggregated_text_feats") / fname
                           for key, fname in text_feat_paths.items()}
        challenge_text_feat_paths = {key: f"aggregated_text_feats/{key}.pickle"
                                     for key in text_feat_paths}
        feature_info = {
            "custom_paths": custom_paths,
            "custom_miech_paths": custom_miech_paths,
            "feature_names": feature_names,
            "subset_list_paths": subset_paths,
            "text_feat_paths": text_feat_paths,
            "challenge_text_feat_paths": challenge_text_feat_paths,
            "raw_captions_path": "raw-captions_marathi.pickle"
        }
        return feature_info

    def load_features(self):
        root_feat = Path(self.root_feat)
        feat_names = {key: self.visual_feat_paths(key) for key in
                      self.paths["feature_names"]}
        custom_path_key = "custom_paths"
        feat_names.update(self.paths[custom_path_key])
        features = {}
        for expert, rel_names in feat_names.items():
            if expert not in self.ordered_experts:
                continue
            feat_paths = tuple([root_feat / rel_name for rel_name in rel_names])
            if len(feat_paths) == 1:
                features[expert] = memcache(feat_paths[0])
            else:
                # support multiple forms of feature (e.g. max and avg pooling). For
                # now, we only support direct concatenation
                msg = f"{expert}: Only direct concatenation of muliple feats is possible"
                print(f"Concatenating aggregates for {expert}....")
                is_concat = self.feat_aggregation[expert]["aggregate"] == "concat"
                self.log_assert(is_concat, msg=msg)
                axis = self.feat_aggregation[expert]["aggregate-axis"]
                x = concat_features.cache_info()  # pylint: disable=no-value-for-parameter
                print(f"concat cache info: {x}")
                features_ = concat_features(feat_paths, axis=axis)
                memory_summary()

                # Make separate feature copies for each split to allow in-place filtering
                features[expert] = copy.deepcopy(features_)

        self.features = features
        if self.challenge_mode:
            self.load_challenge_text_features()
        else:
            self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])
            text_feat_path = root_feat / self.paths["text_feat_paths"][self.text_feat]
            self.text_features = memcache(text_feat_path)

            if self.restrict_train_captions:
                # hash the video names to avoid O(n) lookups in long lists
                train_list = set(self.partition_lists["train"])
                for key, val in self.text_features.items():
                    if key not in train_list:
                        continue

                    if not self.split_name == "full-test":
                        # Note that we do not perform this sanity check for the full-test
                        # split, because the text features in the cached dataset will
                        # already have been cropped to the specified
                        # `resstrict_train_captions`
                        expect = {19, 20}
                        msg = f"expected train text feats as lists with length {expect}"
                        has_expected_feats = isinstance(val, list) and len(val) in expect
                        self.log_assert(has_expected_feats, msg=msg)

                    # restrict to the first N captions (deterministic)
                    self.text_features[key] = val[:self.restrict_train_captions]
        self.summary_stats()

    def sanity_checks(self):
        if self.num_test_captions == 20:
            if len(self.partition_lists["val"]) == 2990:
                missing = 6
            elif len(self.partition_lists["val"]) == 1000:
                missing = 2
            elif len(self.partition_lists["val"]) == 497:
                missing = 0
            else:
                raise ValueError("unrecognised test set")
            msg = "Expected to find two missing queries in MSRVTT for full eval"
            correct_missing = self.query_masks.sum() == self.query_masks.size - missing
            self.log_assert(correct_missing, msg=msg)
