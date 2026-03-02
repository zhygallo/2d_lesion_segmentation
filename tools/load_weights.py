from __future__ import print_function
from __future__ import division

import click
from models.DRUNet32f import get_model


@click.command()
@click.argument('old_weights', type=click.Path(exists=True))
@click.argument('output_weights', type=click.STRING)
@click.option('--img_width', default=240, type=int, help='Image width')
@click.option('--img_height', default=240, type=int, help='Image height')
@click.option('--num_channels', default=2, type=int, help='Number of input channels')
@click.option('--old_num_classes', default=1, type=int, help='Number of classes in old model')
@click.option('--new_num_classes', default=9, type=int, help='Number of classes in new model')
def main(old_weights, output_weights, img_width, img_height, num_channels, old_num_classes, new_num_classes):
    img_shape = (img_width, img_height, num_channels)

    old_model = get_model(img_shape=img_shape, num_classes=old_num_classes)
    old_model.load_weights(old_weights)

    model = get_model(img_shape=img_shape, num_classes=new_num_classes)

    for ind, layer in enumerate(old_model.layers[1:10]):
        if layer.trainable:
            model.layers[ind+1].set_weights(layer.get_weights())

    model.save_weights(output_weights)
    print(f'Saved new weights to {output_weights}')


if __name__ == '__main__':
    main()
