def line2label(line,f,a_dict,o_front,num_zero):
    label = []
    for f_name in f:
        if line[a_dict[f_name]]=='1':
            label.append(1.0)
        else:
            label.append(0.0)
    if o_front:
        return [0 for i in range(num_zero)]+label
    else:
        return label+[0 for i in range(num_zero)]
    
def get_input(txt_dir,f,o_front,num_zero,imfile):
    im_names = []
    domin = []
    label_raw = []
    fea_idx = {}
    lines = open(txt_dir, 'r').readlines()
    all_feas = lines[1].split()
    l = [0,1] if o_front else [1,0]
    
    for i in range(len(all_feas)):
        fea_idx[all_feas[i]] = i
        
    for i in range(2,len(lines)):
        a_line = lines[i]
        im_names.append(imfile+a_line.split()[0])
        domin.append(l)
        label_raw.append(line2label(a_line.split(),f,fea_idx,o_front,num_zero))        
    return im_names,domin,label_raw

def get_inputs(txt1_dir,txt2_dir,f1,f2,imfile1,imfile2):
    im_names1, domin1, label_raw1 = get_input(txt1_dir,f1,False,len(f2),imfile1)
    im_names2, domin2, label_raw2 = get_input(txt2_dir,f2,True,len(f1),imfile2)
    im_names = im_names1 + im_names2
    domin = domin1 + domin2
    label_raw = label_raw1 + label_raw2
    return im_names, domin, label_raw

# get_inputs(self.txt1_dir,self.txt2_dir,self.f1,self.f2,self.imfile1,self.imfile2) 
