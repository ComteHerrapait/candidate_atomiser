import configparser
import paramiko
import scp
import os

def import_data(conf_path="../config/connect.conf"):
    # To change paths, check hdp.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['HDP']['USER']
    password = config['HDP']['PASSWORD']
    ip = config['HDP']['IP_HDP']
    port = config['HDP']['PORT_HDP']
    hdp_path = config['HDP']['PATH_HDFS']
    vm_path = config['HDP']['PATH_VM']
    local_path = config['HDP']['PATH_LOCAL']

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
    # To change paths, check hdp.conf
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['AWS']['USER_NAME']
    hostname = config['AWS']['HOST_NAME']
    key_file = config['AWS']['KEY_FILE']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=user, pkey=key_file)
    stdin, stdout, stderr = ssh.exec_command('ls')


if __name__ == "__main__" : 
    print("running distant parser")
    #import_data()
    export_data()
else :
    print("imported ", __name__)
