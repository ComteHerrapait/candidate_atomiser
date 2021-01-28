import configparser
import paramiko
import scp

def import_data(conf_path="../config/hdp.conf"):
    config = configparser.ConfigParser()
    config.read(conf_path)
    
    user = config['DEFAULT']['USER']
    password = config['DEFAULT']['PASSWORD']
    ip = config['DEFAULT']['IP_HDP']
    port = config['DEFAULT']['PORT_HDP']
    hdp_path = config['DEFAULT']['PATH_HDFS']
    vm_path = config['DEFAULT']['PATH_VM']
    local_path = config['DEFAULT']['PATH_LOCAL']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=port, username=user, password=password)
    stdin, stdout, stderr = ssh.exec_command('mkdir' + vm_path)
    stdin, stdout, stderr = ssh.exec_command('hdfs dfs -get ' + hdp_path + '/data.json ' + vm_path)
    stdin, stdout, stderr = ssh.exec_command('hdfs dfs -get ' + hdp_path + '/categories_string.csv ' + vm_path)
    stdin, stdout, stderr = ssh.exec_command('hdfs dfs -get ' + hdp_path + '/label.csv ' + vm_path)

    lines = stdout.readlines()
    print(lines)

    with scp.SCPClient(ssh.get_transport()) as scp1:
        scp1.get(vm_path + '/data.json')
        scp1.get(vm_path + '/categories_string.csv')
        scp1.get(vm_path + '/label.csv')



def export_data(conf_path="config/conf_aws.json"):
    pass

if __name__ == "__main__" : 
    print("running distant parser")
    import_data()
else :
    print("imported ", __name__)

