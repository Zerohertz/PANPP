import argparse

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description="test global argument parser")

parser.add_argument('--RAW_GT_PATH',help="raw gt path with text files")
parser.add_argument('--RAW_SUBMIT_PATH', help="raw submit path with text files")

# script parameters
parser.add_argument('-g', '--GT_PATH', default='Evaluation_data/gt.zip', help="Path of the Ground Truth file.")
parser.add_argument('-s', '--SUBMIT_PATH', default='Evaluation_data/result.zip', help="Path of your method's results file.")

# webserver parameters
parser.add_argument('-o', '--OUTPUT_PATH', help="Path to a directory where to copy the file that contains per-sample results.")
parser.add_argument('-p', '--PORT', default=8080, help='port number to show')

# result format related parameters
parser.add_argument('--BOX_TYPE', default='QUAD', choices=['LTRB', 'QUAD', 'POLY']) # if type poly check in rrc-evaluation line number 88 split type
parser.add_argument('--TRANSCRIPTION', action='store_true')
parser.add_argument('--CONFIDENCES', action='store_true')
parser.add_argument('--CRLF', action='store_true')

# end-to-end related parameters
parser.add_argument('--E2E', action='store_true')
parser.add_argument('--CASE_SENSITIVE', default=True, type=str2bool)
parser.add_argument('--RS', default=True, type=str2bool)

# evaluation related parameters
parser.add_argument('--AREA_PRECISION_CONSTRAINT', type=float, default=0.5)
parser.add_argument('--GRANULARITY_PENALTY_WEIGHT', type=float, default=1.0)
parser.add_argument('--VERTICAL_ASPECT_RATIO_THRES', default=2.0)

# other parameters
parser.add_argument('-t', '--NUM_WORKERS', default=128, type=int, help='number of threads to use')
parser.add_argument('--GT_SAMPLE_NAME_2_ID', default='([0-9]+)')
parser.add_argument('--DET_SAMPLE_NAME_2_ID', default='([0-9]+)')
parser.add_argument('--PER_SAMPLE_RESULTS', default=True, type=str2bool)

parser.add_argument('--path')

PARAMS = parser.parse_args()
