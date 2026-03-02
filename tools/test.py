import click
import numpy as np
import SimpleITK as sitk
from run_test import get_eval_metrics


@click.command()
@click.argument('mask_file', type=click.Path(exists=True))
@click.argument('pred_mask_file', type=click.Path(exists=True))
def main(mask_file, pred_mask_file):
    mask = sitk.ReadImage(mask_file)
    pred_mask = sitk.ReadImage(pred_mask_file)

    mask = sitk.GetArrayFromImage(mask)
    pred_mask = sitk.GetArrayFromImage(pred_mask)

    dsc, h95, vs = get_eval_metrics(mask, pred_mask)

    print('DSC:', dsc)
    print('H95:', h95)
    print('VS:', vs)


if __name__ == "__main__":
    main()
