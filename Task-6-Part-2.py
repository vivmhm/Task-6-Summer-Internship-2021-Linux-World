#Attaching the EBS Volume to the EC2 Instance that we have just launched
import os
os.system(" aws ec2 attach-volume --volume-id vol-0c1cde20e8e38d7c7 --instance-id i-0f5271540fc179fad --device /dev/sdf")
print("EBS Volume is attached to EC2 Instance successfully")