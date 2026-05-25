# Server for SDAML-Slicer-Plugin

## REST API service

Server support the following commands.

1. `GET /v1/models`  
	Retrieve able models. There are defined in [src/simple_server.py](src/simple_server.py):
	`Resnet`
	`Unet`
	`Unet+Folds`
	`Unet+Folds+TTA`
	`Unet_Aorta`
	`Unet+Folds_Aorta`
	`Unet+Folds+TTA_Aorta`
	
	Output: list of able segmentation models on the server. 

2. `GET /uploads/<name>`  
	Download file <name>from server.  
	
	Output: file under given <name>


3. `GET or POST /`  
	Upload files from form. It is supposed to upload four files with prefixes: `t1_`, `t1c_`, `t2fl_`, `t2_`.
	
	Output: index.html
4. `POST /v1/inference`
    Run autosegmentation.
    
    Output: server response in `plain/text` format
    
## Deployment

1. When first checkout this repo use `git submodule init && git submodule update` to checkout the correspondning version of nnUnet subrepository.

2. Create folders data/input`, `data/output`, `data/tmp` manually, or run the script `mkdir data && mkdir data/input && mkdir data/output && mkdir data/tmp`.

3. Link corresponding trained models to `nUNetv1_data` and `nUNetv2_data` folders:
```
ln -s  tunka.local:/media/storage/pnev/nnUNet_trained_models/ nnUNetv1_data
ln -s  tunka.local:/media/storage/suvorov/nnUNet_results/ nnUNetv2_data
```

4. Use `docker-compose up -d` in this folder to deploy the container. (Give name to the image, f.e. `lapdimo/sdaml-slicer-server:0.1`.)
By default the container will run under `5000` port on the `0.0.0.0` interface.

So user the link to server `http://<server>:5000` in Slicer plugin settings.