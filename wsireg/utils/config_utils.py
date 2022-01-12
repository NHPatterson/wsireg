import yaml


def parse_check_reg_config(yaml_filepath):
    with open(yaml_filepath, "r") as file:
        reg_config = yaml.full_load(file)

    def check_for_key(top_key, check_dict, check_key):
        if check_dict.get(check_key) is None:
            raise ValueError(f"{top_key} does not contain an {check_key}")

    if reg_config.get("project_name") is None:
        raise ValueError(
            f"{yaml_filepath} does not contain a project_name key"
        )

    if reg_config.get("output_dir") is None:
        raise ValueError(f"{yaml_filepath} does not contain a output_dir key")

    if reg_config.get("cache_images") is None:
        reg_config.update({"cache_images": True})

    if reg_config.get("modalities"):
        for key, val in reg_config["modalities"].items():
            [
                check_for_key(key, val, ck)
                for ck in ["image_filepath", "image_res"]
            ]

    if reg_config.get("reg_paths"):

        for key, val in reg_config["reg_paths"].items():
            [
                check_for_key(key, val, ck)
                for ck in [
                    "src_modality_name",
                    "tgt_modality_name",
                    "reg_params",
                ]
            ]
            if isinstance(val.get("reg_params"), str):
                val.update({"reg_params": [val.get("reg_params")]})

    if reg_config.get("attachment_images"):

        for key, val in reg_config["attachment_images"].items():
            [
                check_for_key(key, val, ck)
                for ck in [
                    "attachment_modality",
                    "image_filepath",
                    "image_res",
                ]
            ]

    if reg_config.get("attachment_shapes"):

        for key, val in reg_config["attachment_shapes"].items():
            [
                check_for_key(key, val, ck)
                for ck in ["attachment_modality", "shape_files"]
            ]

    return reg_config
