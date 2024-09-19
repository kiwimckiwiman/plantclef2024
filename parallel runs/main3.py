import subprocess
import os

def main():
    num_processes = 5
    processes = []

    # Path to the directory where sub_script.py is located
    script_directory = r'C:\Users\User\plantclef\please'
    
    # Path to the activation script of your virtual environment
    venv_activate = r'C:\Users\User\plantclef\Scripts\activate'
    # On Unix-based systems, it might be: venv_activate = 'path_to_your_virtualenv/bin/activate'
    
    for i in range(0, num_processes):
        if os.name == 'nt':  # Windows
            # Command to activate the virtual environment and run the script
            command = f'cmd.exe /k "cd /d {script_directory} && {venv_activate} && python mlp_1_64_16_raw.py {i}"'
            process = subprocess.Popen(command, shell=True)
        else:  # Unix-based systems
            # Command to activate the virtual environment and run the script
            command = f'bash -c "cd {script_directory} && source {venv_activate} && python mlp_1_64_16_raw.py {i}"'
            process = subprocess.Popen(['xterm', '-hold', '-e', command])
        
        processes.append(process)

    # Optionally, wait for all processes to complete
    for process in processes:
        process.wait()

if __name__ == "__main__":
    main()
