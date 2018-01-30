import argparse

from evaluation.evaluator import FSNSEvaluator, SVHNEvaluator, TextRecognitionEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that evaluates a trained model on a chosen test set (either FSNS or SVHN)")
    subparsers = parser.add_subparsers(help="choice of evaluation type")
    parser.add_argument("model_dir", help='path to model dir')
    parser.add_argument("snapshot_name", help="name of snapshot in model dir")
    parser.add_argument("eval_gt", help="path to evaluation groundtruth file")
    parser.add_argument("char_map", help="Path to char map")
    parser.add_argument("num_labels", help="number of labels per sample", type=int)
    parser.add_argument("--dropout-ratio", type=float, default=0.5, help="dropout ratio")
    parser.add_argument("--target-shape", default="75,100", help="input shape for recognition network in form: height,width [default: 75,100]")
    parser.add_argument("--timesteps", default=3, type=int, help="number of timesteps localization net shall perform [default: 3]")
    parser.add_argument("--blank-symbol", type=int, default=0, help="blank symbol used for padding [default: 0]")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu to use [default: use cpu]")
    parser.add_argument("--save-rois", action='store_true', default=False, help="save rois of each image for further inspection")
    parser.add_argument("--num-rois", type=int, default=1000, help="number of rois to save [default: 1000]")
    parser.add_argument('--log-name', default='log', help='name of the log file [default: log]')

    fsns_parser = subparsers.add_parser("fsns", help="evaluate fsns model")
    fsns_parser.set_defaults(evaluator=FSNSEvaluator)

    svhn_parser = subparsers.add_parser("svhn", help="evaluate svhn model")
    svhn_parser.set_defaults(evaluator=SVHNEvaluator)

    text_recognition_parser = subparsers.add_parser("textrec", help="evaluate text recognition model")
    text_recognition_parser.set_defaults(evaluator=TextRecognitionEvaluator)

    args = parser.parse_args()
    args.is_original_fsns = True
    args.refinement_steps = 0
    args.refinement = False
    args.render_all_bboxes = False

    evaluator = args.evaluator(args)
    evaluator.evaluate()
