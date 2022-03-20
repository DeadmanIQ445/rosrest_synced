import json


def parse_json(args):
    with open(args['json_path'], 'r') as read_file:
        data = json.load(read_file)

    new_images = [image for image in data['images'] if image['file_name'] == args['file_name']]
    im_id = new_images[0]['id']

    new_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == im_id]

    data['images'] = new_images
    data['annotations'] = new_annotations

    with open(args['new_json_path'], 'w') as file:
        file.write(json.dumps(data))


if __name__ == '__main__':

    args = {
        'json_path': "/home/shamil/PycharmProjects/DetectronAILAB/tmp_images/ground_truth/uchastok_train2021.json",
        'new_json_path': "/home/shamil/PycharmProjects/DetectronAILAB/tmp_images/ground_truth/annotations.json",
        'file_name': "73.tif"
    }

    parse_json(args)

    print('Finished!')
