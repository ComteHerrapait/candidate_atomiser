import configparser
import paramiko
import scp
import os

def import_data(conf_path="../config/connect.conf"):
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
    # To change paths, check connect.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['AWS']['USER']
    hostname = config['AWS']['HOST_NAME_AWS']
    key_file = config['AWS']['KEY_FILE']
    local_path = config['DEFAULT']['PATH_LOCAL_RAW']
    aws_raw_path = config['AWS']['PATH_AWS_RAW']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=user, key_filename=key_file)

    with scp.SCPClient(ssh.get_transport()) as scp1:
        stdin, stdout, stderr = ssh.exec_command('mkdir -p -m=00755 ' + aws_raw_path)

        scp1.put(local_path + 'data.json', aws_raw_path)
        scp1.put(local_path + 'categories_string.csv', aws_raw_path)
        scp1.put(local_path + 'label.csv', aws_raw_path)
        print("Data is now on the VM")

def import_processed_data(conf_path="config/connect.conf"):
    # To change paths, check connect.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['AWS']['USER']
    hostname = config['AWS']['HOST_NAME_AWS']
    key_file = config['AWS']['KEY_FILE']
    local_path = config['DEFAULT']['PATH_LOCAL_PROCESSED']
    aws_processed_path = config['AWS']['PATH_AWS_PROCESSED']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=user, key_filename=key_file)

    with scp.SCPClient(ssh.get_transport()) as scp1:
        if not os.path.exists(local_path):
            os.makedirs(local_path)
            print("Local directory created")
        
        scp1.get(aws_processed_path + 'result.csv', local_path)
        print("Processed data is now on your machine")


if __name__ == "__main__" : 
    print("running distant parser")
    #import_data()
    #export_data()
    import_processed_data()
else :
    print("imported ", __name__)
