######################################################################
#  script name  : fileSynchronizer.py
#  author       : Lcx
#  created time : 2019.7.20
#  modification time ：2019.7.21
######################################################################

import paramiko,os
import hashlib
# ssh传输类：

class fileUploaderClass(object):
    def __init__(self,serverIp,userName,passWd,port=22):
        self.__ip__         = serverIp
        self.__userName__   = userName
        self.__passWd__     = passWd
        self.__port__       = port
        self.__ssh__        = paramiko.SSHClient()
        self.__ssh__.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def sshScpPut(self,localFile,remoteFile):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        remoteDir  = remoteFile.split("/")
        if remoteFile[0]=='/':
            sftp.chdir('/')
            
        for item in remoteDir[0:-1]:
            if item == "":
                continue
            try:
                sftp.chdir(item)
            except:
                sftp.mkdir(item)
                sftp.chdir(item)
        sftp.put(localFile,remoteDir[-1])
        sftp.close()
        self.__ssh__.close()
        print("ssh localfile:%s remotefile:%s success"%(localFile,remoteFile))

    def sshScpGetNames(self,remoteDir):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        wo = sftp.listdir(remoteDir)
        return wo
    
    def sshScpGet(self, remoteFile, localFile, showProgress=False):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        try:
            sftp.stat(remoteFile)
            print("Remote file exists!")
        except:
            print("Remote file does not exist!")
            return False
        sftp = self.__ssh__.open_sftp()
        if showProgress:
            sftp.get(remoteFile, localFile,callback=self.__putCallBack__)
        else:
            sftp.get(remoteFile, localFile)
        sftp.close()
        self.__ssh__.close()
        return True
    
    def __putCallBack__(self,transferred,total):
        print("current transferred %3.1f percent"%(transferred/total*100),end='\r')
    
    def sshScpGetmd5(self, file_path):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp() 
        try:
            file = sftp.open(file_path, 'rb')
            res  = (True,hashlib.new('md5', file.read()).hexdigest())
            sftp.close()
            self.__ssh__.close()
            return res
        except:
            sftp.close()
            self.__ssh__.close()
            return (False,None)
    def sshScpRename(self, oldpath, newpath):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.rename(oldpath,newpath)
        sftp.close()
        self.__ssh__.close()
        print("ssh oldpath:%s newpath:%s success"%(oldpath,newpath))

    def sshScpDelete(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.remove(path)
        sftp.close()
        self.__ssh__.close()
        print("ssh delete:%s success"%(path))
    
    def sshScpDeleteDir(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        self.__rm__(sftp,path)
        sftp.close()
        self.__ssh__.close()
        
    def __rm__(self,sftp,path):
        try:
            files = sftp.listdir(path=path)
            print(files)
            for f in files:
                filepath = os.path.join(path, f).replace('\\','/')
                self.__rm__(sftp,filepath)
            sftp.rmdir(path)
            print("ssh delete:%s success"%(path))
        except:
            print(path)
            sftp.remove(path)
            print("ssh delete:%s success"%(path))

if __name__ == "__main__":
    basePath    = "/home/gdp/CXH/StyleTransfer/train_logs/SN2/"
    uploader    = fileUploaderClass("192.168.101.57","gdp","glass123456")
    remote      = os.path.join(basePath,"model_config11.json").replace('\\','/')
    local       = "./model_config.json"
    uploader.sshScpGet(remote,local,True)



