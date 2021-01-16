from subprocess import call 

models_dict = {'citypersons_cascadeHRnet': 'epoch_5.pth.stu', 
	'caltech_cascadeHRnet': 'epoch_14.pth.stu', 
	'eurocity_cascadeHRnet': 'epoch_147.pth.stu', 
	'crowdhuman1_cascadeHRnet': 'epoch_34.pth.stu', 
	'crowdhuman2_cascadeHRnet': 'epoch_19.pth.stu', 
	'widerpedestrian_cascadeHRnet': 'epoch_14.pth',
    'citypersons_fasterRCNN': 'epoch_1.pth.stu', 
    'eurocity_fasterRCNN': 'epoch_35.pth.stu', 
    'citypersons_retinanet': 'epoch_7.pth.stu', 
    'citypersons_retinanetguided': 'epoch_72.pth.stu', 
    'citypersons_hybridTaskCascade': 'epoch_6.pth.stu', 
    'citypersons_MGAN': 'epoch_1.pth'}

#returnval = call('python tools/demo.py configs/elephant/cityperson/cascade_hrnet.py ./models_pretrained/epoch_5.pth.stu images/test/ result_our_cityperson_cascade_hr/',  shell=True)
#returnval = call('python tools/demo.py configs/elephant/caltech/cascade_hrnet.py ./models_pretrained/epoch_14.pth.stu images/test/ result_our_caltech_cascade_hr/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/eurocity/cascade_hrnet.py ./models_pretrained/epoch_147.pth.stu images/test/ result_our_eurocity_cascade_hr/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/crowdhuman/cascade_hrnet.py ./models_pretrained/epoch_34.pth.stu images/test/ result_our_crowdhuman1_cascade_hr/', shell=True)
returnval = call('python tools/demo.py configs/elephant/crowdhuman/cascade_hrnet.py ./models_pretrained/epoch_19.pth.stu demo/ final_results/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/wider_pedestrain/cascade_hrnet.py ./models_pretrained/epoch_14.pth images/test/ result_our_widerpedestrian_cascade_hr/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/eurocity/faster_rcnn_hrnet.py ./models_pretrained/epoch_35.pth.stu images/test/ result_our_eurocity_faster_RCNN/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/cityperson/retinanet_ResNeXt101.py ./models_pretrained/epoch_7.pth.stu images/test/ result_our_citypersons_retinanet/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/cityperson/htc_ResNeXt101.py ./models_pretrained/epoch_6.pth.stu images/test/ result_our_citypersons_hybridTaskCascade/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/cityperson/mgan_vgg.py ./models_pretrained/epoch_1.pth images/test/ result_our_citypersons_MGAN/', shell=True)
#returnval = call('python tools/demo.py configs/elephant/cityperson/faster_rcnn_hrnet.py ./models_pretrained/epoch_1.pth.stu images/test/ result_our_citypersons_fasterRCNN/', shell=True)

