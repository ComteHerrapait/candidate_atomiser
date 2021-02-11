import configparser
import paramiko
import scp
import os
import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet, InvalidToken

def import_data(conf_path="../config/connect.conf"):
    """Imports data from Hadoop Virtual Machine. Loading variables that are in connect.conf file.
    It creates local directory to store data before sending them to AWS VM.

    Args:
        conf_path (str, optional): [path to the configuration file]. Defaults to "../config/connect.conf".
    """    
    # To change paths, check connect.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['HDP']['USER']
    password = config['HDP']['PASSWORD']
    ip = config['HDP']['HOST_NAME_HDP']
    port = config['HDP']['PORT_HDP']
    hdp_path = config['HDP']['PATH_HDFS']
    vm_path = config['HDP']['PATH_VM']
    local_path = config['DEFAULT']['PATH_LOCAL_RAW']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=port, username=user, password=password)
    stdin, stdout, stderr = ssh.exec_command('mkdir' + vm_path)
    stdin, stdout, stderr = ssh.exec_command('hdfs dfs -get ' + hdp_path + '/data.json ' + vm_path)
    stdin, stdout, stderr = ssh.exec_command('hdfs dfs -get ' + hdp_path + '/categories_string.csv ' + vm_path)
    stdin, stdout, stderr = ssh.exec_command('hdfs dfs -get ' + hdp_path + '/label.csv ' + vm_path)

    print("Data succesfully retrieved on the virtual machine")

    with scp.SCPClient(ssh.get_transport()) as scp1:
        if not os.path.exists(local_path):
            os.makedirs(local_path)
            print("Local directory created")
        
        scp1.get(vm_path + '/data.json', local_path)
        scp1.get(vm_path + '/categories_string.csv', local_path)
        scp1.get(vm_path + '/label.csv', local_path)
        print("Data is locally here")



def export_data(conf_path="config/connect.conf"):
    """Exports data from the local machine to AWS. Loading variables that are in connect.conf file.
    It creates directories on the VM to store data before processing. Also encrypts data.

    Args:
        conf_path (str, optional): [path to configuration file]. Defaults to "config/connect.conf".
    """    

    # To change paths, check connect.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['AWS']['USER']
    hostname = config['AWS']['HOST_NAME_AWS']
    key_file = config['AWS']['KEY_FILE']
    local_path = config['DEFAULT']['PATH_LOCAL_RAW']
    aws_raw_path = config['AWS']['PATH_AWS_RAW']


    # Generate key
    config.read("config/key.conf") # Configuration file to generate the key. Change path if needed
    password = config['DEFAULT']['CRYPTING_KEY'] # Useful for the key to be the same everywhere
    password = password.encode()
    salt = bytes(config['DEFAULT']['SALT'], encoding='utf-8')  # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))

    # Encrypt data.json

    with open(local_path + 'data.json', 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    encrypted = fernet.encrypt(data)

    with open(local_path + 'encrypted_data.json', 'wb') as f:
        f.write(encrypted)

    # Encrypt label.csv
    with open(local_path + 'label.csv', 'rb') as f:
        data = f.read()

    encrypted = fernet.encrypt(data)

    with open(local_path + 'encrypted_label.csv', 'wb') as f:
        f.write(encrypted)
    
    # Encrypt categories_string.csv
    with open(local_path + 'categories_string.csv', 'rb') as f:
        data = f.read()

    encrypted = fernet.encrypt(data)

    with open(local_path + 'encrypted_categories_string.csv', 'wb') as f:
        f.write(encrypted)
    

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=user, key_filename=key_file)

    with scp.SCPClient(ssh.get_transport()) as scp1:
        stdin, stdout, stderr = ssh.exec_command('mkdir -p -m=00755 ' + aws_raw_path)
        scp1.put(local_path + 'encrypted_data.json', aws_raw_path)
        scp1.put(local_path + 'encrypted_categories_string.csv', aws_raw_path)
        scp1.put(local_path + 'encrypted_label.csv', aws_raw_path)
        scp1.put("config/key.conf", aws_raw_path)
        print("Data is now on the VM")

def import_processed_data(conf_path="config/connect.conf"):
    """Imports processed data from AWS VM. Decrypts data from result.csv.
    It creates local directory to store data before sending them to AWS VM.
    Args:
        conf_path (str, optional): [path to configuration file]. Defaults to "config/connect.conf".
    """   

    # To change paths, check connect.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['AWS']['USER']
    hostname = config['AWS']['HOST_NAME_AWS']
    key_file = config['AWS']['KEY_FILE']
    local_path = config['DEFAULT']['PATH_LOCAL_PROCESSED']
    aws_processed_path = config['AWS']['PATH_AWS_PROCESSED']
    aws_raw_path = config['AWS']['PATH_AWS_RAW']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=user, key_filename=key_file)

    with scp.SCPClient(ssh.get_transport()) as scp1:
        if not os.path.exists(local_path):
            os.makedirs(local_path)
            print("Local directory created")
        
        scp1.get(aws_processed_path + 'result.csv', local_path)
        print("Processed data is now on your machine")

    # Decrypt result.csv
    config.read("config/key.conf")
    password = config['DEFAULT']['CRYPTING_KEY'] # Useful for the key to be the same everywhere
    password = password.encode()
    salt = bytes(config['DEFAULT']['SALT'], encoding='utf-8')
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))

    input_file = 'result.csv'
    output_file = 'decrypted_result.csv'

    with open(local_path + input_file, 'rb') as f:
        data = f.read()  # Read the bytes of the encrypted file

    fernet = Fernet(key)
    try:
        decrypted = fernet.decrypt(data)

        if not os.path.exists(local_path + output_file):
            open(local_path + output_file, 'w').close()

        with open(local_path + output_file, 'wb') as f:
            f.write(decrypted)  # Write the decrypted bytes to the output file

    os.remove(local_path + input_file)

    except InvalidToken as e:
        print("Invalid Key - Unsuccessfully decrypted")


if __name__ == "__main__" : 
    print("running distant parser")
    #import_data()
    export_data()
    import_processed_data()
else :
    print("imported ", __name__)
