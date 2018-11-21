Follow carefully and with detail:
sudo apt-get install linux-generic-lts-xenial (Not sure if it matters from https://github.com/ApolloAuto/apollo/blob/c8e7d53e8e364315727d535b66b2f76dda33bf2b/docs/specs/Software_and_Kernel_Installation_guide.md)

https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/apollo_software_installation_guide.md

--> Restart linux and before login run CTR + F1 ->
sudo service lightdm stop (Or whatever x server you have)
sudo init 3 

--> blacklist naouveau
nano /etc/modprobe.d/blacklist-nouveau.conf
# Write in file: 
blacklist nouveau
options nouveau modeset=0
sudo update-initramfs -u # Regenerate the kernel initramfs


================================================================================

sudo apt-get install sshuttle

================================================================================