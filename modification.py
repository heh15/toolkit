'''
By Hao He, Jan 10th, 2020

'''

import os, sys, glob
from pathlib import Path
import copy
import re
from subprocess import call, Popen, PIPE
import numpy as np
import pandas as pd
import datetime

### find the directory for the file
def find_targets(filename, directory, dimensions=0, subfolders='all', cwd=True):
    '''
    To do: multiple levels of subfolders. 
    Jan. 20th, can search for one level of subfolders. 
    '''
    if type(dimensions) == int:
        if subfolders=='all':
            targets=list(Path(directory).rglob(filename))
        else:              
            for f in subfolders:
                p=Popen(['find', directory, '-maxdepth',str(dimensions), '-mindepth', str(dimensions),'-name', f], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err=p.communicate()
            directories=output.decode('utf-8').split('\n')
            directories.remove('')
            directories=[f+'/' for f in directories]
            targets=[]
            for directory in directories:
                target=list(Path(directory).rglob(filename))
                targets=targets+target

    return targets

def find_others(targets, cwd=True, origfile=''):
    '''
    To do: 
    1) try to match '/1/home/heh15/...' with '/home/heh15/...'
    '''
    filename=os.path.basename(targets[0])
    if cwd==True:
        origfile= Path(os.getcwd()+'/'+filename)
    else:
        origfile=Path(origfile)
    targets_other=targets.copy()
    targets_other.pop(targets.index(origfile))

    return origfile, targets_other
  
def write_targets(targets, folder='.'):
    '''
    Feb. 11th, 2020
    1) Name the script with current time. 
    '''
    now=datetime.datetime.today()
    scriptName=now.strftime("%Y%m%d-%H%M%S")
    # scriptName=os.path.basename(targets[0])
    # scriptName='.'.join(scriptName.split('.')[:-1])
    scriptFile=scriptName+'_copy.sh'
    with open(scriptFile,'w') as output:
        for target in targets:
            output.write('scp -r heh15@laburnum:'+str(target)+' '+folder+' \n')
  
### copy the modification into similar files.
def _extract_txt(content, linerange, section, private, mode):
    ''' only used inside the module
    To do: 
    1) Need to comment out lines the same way as the emacs way. 
    '''
    content_pd=pd.DataFrame(content, columns=['content'])
    content_pd['row number']=content_pd.index
    content_pd['private']=False    
    # start from the '# main program'. 
    startindexes=content_pd.loc[content_pd['content'].str.contains(section, regex=False)].reset_index(drop=True)
    if len(startindexes)!=0:
        startline=startindexes['row number'][0]-1
        content_mod_pd=content_pd.loc[content_pd['row number']>=startline]
    else:
        startindexes=content_pd.loc[content_pd['content'].str.contains('main program', regex=False)].reset_index(drop=True)
        startline=startindexes['row number'][0]-1
        newline= pd.DataFrame({"content": "#"*60, "row number": startline}, index=[(startline)])
        content_pd['row number'].iloc[(startline):]=content_pd['row number'].iloc[(startline):]+1
        content_pd=pd.concat([content_pd.iloc[:(startline)], newline, content_pd.iloc[(startline):]]).reset_index(drop=True)
        content_mod_pd=content_pd.loc[content_pd['row number']>=startline]
    stopindexes=content_mod_pd.loc[content_mod_pd['content'].str.contains('\#{60}', regex=True)].reset_index(drop=True)
    if len(stopindexes)>1:
        stopline=stopindexes['row number'][stopindexes.index[1]]
        content_mod_pd=content_mod_pd.loc[content_mod_pd['row number']<stopline]
    content_mod_pd=content_mod_pd.loc[content_mod_pd['row number']>=linerange[0]]
    if linerange[1]!=-1:
        content_mod_pd=content_mod_pd.loc[content_mod_pd['row number']<=linerange[1]]
    # select private regions.     
    private_start=content_pd.loc[content_pd['content'].str.contains('\#{10} private region', regex=True)].reset_index(drop=True)['row number']
    private_stop=content_pd.loc[content_pd['content'].str.match(r'\#{20}(?!#).*')].reset_index(drop=True)['row number']
    for i in range(len(private_start)):
        content_pd['private'].iloc[(private_start[i]+1):private_stop[i]]=True
    content_mod_pd['private'].loc[content_pd['private']==True]=True
    if mode=='input':
        if private==True:
            condition1=content_mod_pd['private']==True
            condition2=content_mod_pd['content'].str.match(r'^[\t ]*#')
            condition= condition1 & condition2
            content_mod_pd['content'].loc[condition]=content_mod_pd['content'].loc[condition].str.replace('# ','',1, regex=True)

    # # remove space lines.
    # blankline=content_mod_pd['content'].str.match(r'^(?:[\t ]*(?:\r?\n|\r))+ ')
    # content_mod_pd=content_mod_pd.loc[~blankline]
    # # remove the file start with #.
    # commentline=content_mod_pd['content'].str.match(r'^[\t ]*#.*$')
    # content_mod_pd=content_mod_pd.loc[~commentline]

    # # remove the line containing filename
    # fileline=content_mod_pd[0].str.match(r'^.*= *.*\+?[\'\"].*[\'\"]$')
    # content_mod_pd=content_mod_pd.loc[~fileline]
    # # remove the protected regions. 
    # protected_start=content_pd.loc[content_pd[0].str.contains('main program', regex=False)].index[0]
       # protected_stop=content

    content_mod_pd=content_mod_pd.reset_index(drop=True)

    return content_pd, content_mod_pd

def _group_private(content_mod_pd):

    adj_check=(content_mod_pd['private'] != content_mod_pd['private'].shift()).cumsum()
    private_condition=content_mod_pd['private']==True
    private_groupfunc=content_mod_pd.loc[private_condition].groupby(adj_check, as_index=False, sort=False)
    private_group=private_groupfunc.agg({'content': 'sum',
                                          'row number': ['first','last']})

    return private_group



def copy_modification(origfile, targets_other, inputrange=[0, -1], outputrange=[0, -1], section='main program', append=False, private=True, **kwargs):
    '''
    copy the modification of one file to duplicate files.
    Feb. 4th, 2020, 
    1) can comment out private regions while moving it to other scripts.
    Jan. 31st, 2020, 
    1) can append several lines to the end of the assigned section. 
    Jan. 24th, 2020, can copy any section and add section to new files. 
    Jan. 17th, 2020, can substitue from start line to stop line. 
    Jan. 12th, 2020, only substitute main program part. 
    '''
    with open(origfile, 'r') as infile:
        content=infile.readlines()
    content_mod_pd=_extract_txt(content, inputrange, section, private, 'input')[1]
    private_group1=_group_private(content_mod_pd)

    for target in targets_other:
        with open (target, 'r') as outfile:
            content_out=outfile.readlines()
        content_out_pd, content_out_mod_pd=_extract_txt(content_out, outputrange, section, private, 'output')

        private_group2=_group_private(content_out_mod_pd)
        if append==True:
            startrow=content_out_mod_pd['row number'].iloc[-1]
            content_out=content_out_pd['content'].to_list()
        else:
            startrow=content_out_mod_pd['row number'][0]
            content_out_basic=content_out_pd.append(content_out_mod_pd, ignore_index=True)
            content_out_basic=pd.concat([content_out_pd, content_out_mod_pd]).drop_duplicates(subset='row number', keep=False)
            content_out=content_out_basic['content'].to_list()
        matched=private_group1['content'].isin(private_group2['content'])
        private_startlines=private_group1.loc[~matched['sum']]['row number']['first'].to_numpy()
        private_stoplines=private_group1.loc[~matched['sum']]['row number']['last'].to_numpy()+1
        private_ranges=np.transpose(np.vstack([private_startlines, private_stoplines]))
        if len(private_ranges) ==0:
            content_mod=content_mod_pd['content'].tolist()
        else:
            for private_range in private_ranges:
                condition=(content_mod_pd['row number'] >=private_range[0]) & (content_mod_pd['row number'] < private_range[1])
                content_mod_pd['content'].loc[condition]='# '+content_mod_pd['content'].loc[condition]
            content_mod=content_mod_pd['content'].tolist()
            content_mod_pd['content'].loc[condition]=content_mod_pd['content'].loc[condition].str.replace('# ','',1, regex=True)
        row=startrow
        for line in content_mod:
            content_out.insert(row, line)
            row=row+1
        with open (target, 'w') as outfile:
            for line in content_out:
                outfile.write(line)
        
    return targets_other

### execute the script
def execute_targets(targets, other=False):
    '''
    To do:
    Feb. 4th, 2020
    1) change the command from 'exec' to 'os.system', scripts are properly executed.  
    '''
    if other==True:
        targets=find_others(targets)[1]
    files_error=[]
    for target in targets:
        try:
            os.system('python '+str(target))
        except:
            files_error.append(target)
    for target in files_error:
        print(str(target)+' \n')
        
    return files_error
           
### create script in the directory
def create_targets(origfile, directory, subfolders='all'):
    if subfolders=='all':
        subdirectories=[f.path for f in os.scandir(directory) if f.is_dir()]
    else:
        subdirectories=[directory+f for f in subfolders]
    for f in subdirectories:
        if f in str(origfile):
            followpath=str(origfile).replace(f, "")
    paths=[]
    for f in subdirectories:
        if (f in str(origfile))==False:
            path=re.sub(r'/((?!/).)*?$', '/', (f+followpath))
            filename=(re.search(r'/((?!/).)*?$',(f+followpath)).group()).strip('/')
            paths.append(f+followpath)
            call(['mkdir', '-p', path])
            call(['cp', str(origfile), path])
            print(f+followpath+'\n')

    return paths
    
def create_targets_nd(origfile, directory, subfolders='all', dimensions=0):
    ''' 
    Feb. 28th, fully modified and tested the function of creating certain files in multiple subfolders. 
    Jan. 11th, complete the function, partially tested. It can be applied to both files and directories. 
    Jan. 10th, 2020, still under construction, haven't tested yet
    '''
    origfile_split=str(origfile).split('/')
    if type(dimensions) == int:
        index=len(directory.split('/'))+dimensions-1
        if subfolders=='all':
            subdirectory='/'.join(origfile_split[0:index])+'/'
            subdirectories=[f.path for f in os.scandir(subdirectory) if f.is_dir()]
            subfolders=[f.split('/')[-1] for f in subdirectories]
        paths=[]
        for f in subfolders:
            origfile_split_cp=origfile_split.copy()
            origfile_split_cp[index]=f
            paths.append('/'.join(origfile_split_cp))      
    if type(dimensions) == list:
        # indexes=len(directory.split('/'))+np.array(dimensions)-1
        paths=[]
        for dimension in dimensions:
            folders=subfolders[dimension]
            index=len(directory.split('/'))+dimension-1
            if folders=='all':
                subdirectory='/'.join(origfile_split[0:index])+'/'
                subdirectories=[f.path for f in os.scandir(subdirectory) if f.is_dir()]
                folders=[f.split('/')[-1] for f in subdirectories]
            for f in folders:
                origfile_split_cp=origfile_split.copy()
                origfile_split_cp[index]=f
                paths.append('/'.join(origfile_split_cp))
    paths_other=paths.copy()
    if str(origfile) in paths_other:
        paths_other.remove(str(origfile))
    if origfile.is_dir():
        for path in paths_other:
           call(['mkdir', '-p', path])
           print(path+'\n')
    if origfile.is_file():
        for path in paths_other:
            subdirectory=re.sub(r'/((?!/).)*?$', '/', path)
            call(['mkdir', '-p', subdirectory])
            call(['cp', str(origfile), path])
            print(path+'\n')

    return paths
