test_reconstruction:
	./test_reconstruction.py \
		--loader-config configs/nrrd_loader.yaml \
		--generator-config configs/reconstruction.yaml

test_block_generator_parsing_loader:
	./sample_generator.py \
		--loader-config configs/parsing_loader.yaml \
		--generator-config configs/block_generator.yaml

test_block_generator_nrrd_loader:
	./sample_generator.py \
		--loader-config configs/nrrd_loader.yaml \
		--generator-config configs/block_generator.yaml

test_block_generator_nifti_loader:
	./sample_generator.py \
		--loader-config configs/nifti_loader.yaml \
		--generator-config configs/block_generator.yaml

test_block_sampler_parsing_loader:
	./sample_generator.py \
		--loader-config configs/parsing_loader.yaml \
		--generator-config configs/block_sampler.yaml

test_block_sampler_nrrd_loader:
	./sample_generator.py \
		--loader-config configs/nrrd_loader.yaml \
		--generator-config configs/block_sampler.yaml

test_block_sampler_nifti_loader:
	./sample_generator.py \
		--loader-config configs/nifti_loader.yaml \
		--generator-config configs/block_sampler.yaml

test_nifti_loader:
	./sample_loader.py \
		--loader-config configs/nifti_loader.yaml

test_parsing_loader:
	./sample_loader.py \
		--loader-config configs/parsing_loader.yaml

test_nrrd_loader:
	./sample_loader.py \
		--loader-config configs/nrrd_loader.yaml
