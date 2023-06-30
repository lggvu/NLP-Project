from nltk.translate.bleu_score import corpus_bleu
import argparse

def evaluate(results, reference_labels):
    with open(results, "r") as f:
        results = f.readlines()
    with open(reference_labels, "r") as f:
        reference_labels = f.readlines()
    

    # results = ["Anh chàng trai của Albert Barnett và người vợ, Susan Barnett, từ khu vực Đông Phi tại Tussoo, Alabama"]
    # reference_labels = ["Anh Albert Barnett và chị Susan Barnett , thuộc hội thánh West ở Tuscaloosa , Alabama"]

    # Preprocess the reference labels
    reference_labels = [label.strip().split() for label in reference_labels]

    # Preprocess the generated results
    generated_results = [result.split() for result in results]

    # Calculate the BLEU score
    bleu_score = corpus_bleu([[label] for label in reference_labels], generated_results, weights=[1/4,1/4,1/4,1/4])
    print("BLEU:", bleu_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--ref_file", type=str, help="reference labels")


    args = parser.parse_args()
    args.output_file="/home2/khanhnd/transformer-mt/output_txt.txt"
    args.ref_file="/home2/khanhnd/fairseq/examples/translation/pho-mt-backtranslation/test.vi"
    evaluate(results=args.output_file, reference_labels=args.ref_file)
    
