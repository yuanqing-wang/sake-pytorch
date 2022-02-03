def run():
    import os
    os.system('git clone https://github.com/vgsatorras/en_flows.git')
    file_handle = open("en_flows/dw4_experiment/dataset.py", "r")
    lines = file_handle.readlines()
    lines = [line.replace("dw4_experiment/data/", "en_flows/dw4_experiment/data/") for line in lines]
    file_handle.close()
    file_handle = open("en_flows/dw4_experiment/dataset.py", "w")
    file_handle.writelines(lines)
    file_handle.close()


if __name__ == "__main__":
    run()
