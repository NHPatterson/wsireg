import yaml
yaml_filepath = "C:/pyproj/wsireg/wsireg/reg_templates/basic_template.yaml"

def parse_wsireg_config(yaml_filepath):
    with open(yaml_filepath, "r") as file:
        reg_config = yaml.safe_load(file)
