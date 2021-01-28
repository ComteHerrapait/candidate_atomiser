import configparser
import paramiko

def import_data(conf_path="config/hdp.conf"):
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
    ssh.connect(ip+port, user, password)
    stdin, stdout, stderr = ssh.exec_command('ls')
    lines = stdout.readlines()
    print(lines)



def export_data(conf_path="config/conf_aws.json"):
    pass

if __name__ == "__main__" : 
    print("running distant parser")
    import_data()
else :
    print("imported ", __name__)

