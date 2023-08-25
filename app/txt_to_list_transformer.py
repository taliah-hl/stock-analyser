# usage:
# run this file to change txt file to list
# so we can get a list of stock to put in config file

txt_file='../hi/trylst.txt'  # put txt file name here (file has to be in app folder)

# e.g. file in 1 layer upper in directory
# txt_file='../hotstock25.txt'


with open(txt_file, 'r') as fio:
    lines = fio.readlines()
        
lines = [i.strip(" []\n") for i in lines]


print(lines)        # will print the list in console for copying