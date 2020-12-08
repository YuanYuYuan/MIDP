check_sampler_quality:
	./check_sampler_quality.py \
		--loader-config configs/loader_abcs_task3.yaml \
		--generator-config configs/genenrator_abcs_fg_bg.yaml \

# postponed
test_augmentation_affine:
	rm -rvf outputs
	./sample_generator.py \
		--loader-config configs/loader_abcs.yaml \
		--generator-config configs/generator_abcs.yaml \
		--output-dir outputs

test_generator_abcs_optic_bbox:
	rm -rvf outputs
	./sample_generator.py \
		--loader-config configs/loader_abcs_task2_optic_bbox.yaml \
		--generator-config configs/genenrator_abcs_optic_bbox.yaml \
		--output-dir outputs

test_reconstruction_abcs_optic_bbox:
	./reconstruction_with_reverter.py \
		--loader-config configs/loader_abcs_task2_optic_bbox.yaml \
		--generator-config configs/genenrator_for_reconstruction_abcs_optic_bbox.yaml

find_bounding_box_ABCs_optic:
	./find_box.py \
		--config configs/loader_abcs_task2_optic.yaml \
		--output bbox.json \
		--padding 20

test_reconstruction_with_flat_gen:
	./reconstruction_with_reverter.py \
		--loader-config configs/loader_abcs.yaml \
		--generator-config configs/generator_for_flat_recon.yaml

test_flat_gen:
	rm -rvf outputs
	./sample_generator.py \
		--loader-config configs/loader_abcs.yaml \
		--generator-config configs/generator_for_flat.yaml \
		--output-dir outputs

test_abcs_2d_pipeline:
	./reconstruction_with_reverter.py \
		--loader-config configs/loader_abcs.yaml \
		--generator-config configs/abcs_2d_generator.yaml \
		--output-dir outputs

test_block_generator_for_detector:
	./generator_for_detector.py \
		--loader-config configs/nifti_loader.yaml \
		--generator-config configs/generator_for_detector.yaml \
		--output-dir outputs

find_bounding_box_ABCs:
	./find_box.py \
		--config configs/nifti_loader_ABCs_vis.yaml \
		--output bbox.json \
		--padding 20

test_augmentation:
	rm -rvf outputs
	./sample_generator.py \
		--loader-config configs/nifti_loader_test_aug.yaml \
		--generator-config configs/genenrator_test_aug.yaml \
		--output-dir outputs

nrrd_to_nifti_vis:
	./nrrd2nifti.py \
		--config configs/nrrd2nifti_vis.yaml

test_block_generator_bbox:
	./sample_generator.py \
		--loader-config configs/nrrd_loader_eyes_bbox.yaml \
		--generator-config configs/block_generator_bbox.yaml \
		--output-dir outputs

test_nrrd_loader_with_bbox:
	./sample_loader.py \
		--loader-config configs/nrrd_loader_eyes_bbox.yaml

test_block_generator_msd_loader:
	./sample_generator.py \
		--loader-config configs/msd_loader.yaml \
		--generator-config configs/block_generator.yaml

test_msd_loader:
	./sample_loader.py \
		--loader-config configs/msd_loader.yaml

test_reconstruction_with_reverter_2d:
	./reconstruction_with_reverter.py \
		--loader-config configs/nrrd_loader.yaml \
		--generator-config configs/reconstruction_2d.yaml


resample_cyberknife_dataset:
	./resample_cyberknife_dataset.py \
		--data-dir cyberknife \
		--spacing 1 \
		--output-dir resampled_data

split_datalist_train_valid_test:
	./generate_data_list.py \
		--loader-config configs/nrrd_loader.yaml \
		--output train_valid_test_list.yaml \
		--split-ratio 0.6 0.2

resample_nrrd_dataset:
	./resample_nrrd_dataset.py \
		--data-dir data \
		--spacing 1 \
		--output-dir resampled_data

find_bounding_box:
	./find_box.py \
		--config configs/nrrd_loader_eyes.yaml \
		--output bbox.json \
		--padding 20

find_optic_bbox:
	./find_box.py \
		--config configs/train_list_eyes.yaml \
		--output optic_bbox.json

test_saving_resample_prediction:
	./save_resample_prediction.py \
		--loader-config configs/nrrd_loader_resample.yaml

test_block_generator_nrrd_loader_resample:
	./sample_generator.py \
		--loader-config configs/nrrd_loader_resample.yaml \
		--generator-config configs/block_generator.yaml

test_block_generator_crop_nrrd_loader:
	./sample_generator.py \
		--loader-config configs/nrrd_loader.yaml \
		--generator-config configs/block_generator_crop.yaml

test_reconstruction:
	./reconstruction.py \
		--loader-config configs/nrrd_loader.yaml \
		--generator-config configs/reconstruction.yaml

test_reconstruction_with_reverter:
	./reconstruction_with_reverter.py \
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
