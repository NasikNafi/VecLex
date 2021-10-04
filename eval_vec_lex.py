import os,glob
import argparse
import torch
from veclex_metric import compute_veclex 

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)


def main(args):
    decoded_path = os.path.join(args.summary_dir, '')
    all_scores = []
    for reference_file in glob.glob(os.path.join(args.reference_dir, '*.ref')):
        filename = reference_file[reference_file.rfind('/')+1:-4]
        decoded_file = decoded_path + filename + '.dec'

        with open(reference_file, 'r') as rf:
            reference_text = rf.read()

            with open(decoded_file, 'r') as df:
                decoded_text = df.read()
                similarity_score = compute_veclex(decoded_text.split(), reference_text.split())
                all_scores.append(similarity_score)

    avg_score = sum(all_scores) / len(all_scores)
    print('Total number of summaries: ', len(all_scores))
    print('Average VecLex score: ', avg_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files based on VecLex similarity')

    parser.add_argument('--reference_dir', action='store', required=True,
                        help='directory of reference/ground truth summaries')
    parser.add_argument('--summary_dir', action='store', required=True,
                        help='directory of decoded/generated summaries')
    args = parser.parse_args()
    main(args)
