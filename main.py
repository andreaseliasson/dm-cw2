from pre_process import *
from bag_of_words import *

if __name__ == "__main__":
    parse_html = False
    if parse_html:
        sub_dirs = get_sub_dir('../data/gap-html')
        text_docs = get_html_files_from_sub_dirs('../data/gap-html', sub_dirs)
        raw_docs = parse_html_files(text_docs)
        to_csv(raw_docs)
    else:
        raw_docs_df = pd.read_csv('../data/raw_text.csv', usecols=['id', 'raw_text'])
        raw_text_list = raw_docs_df['raw_text'].values.tolist()
        print(raw_text_list[0][:10])
        print(len(raw_text_list[0]))
        print(len(raw_text_list))

        tokenized_text = tokenize_remove_punct(raw_text_list[0])
        filter_tokenized_text = filter_words(tokenized_text)
        filter_tokenized_text_non_short = remove_short_words(filter_tokenized_text)

        stemmed_text = stem_porter(filter_tokenized_text_non_short)
        print(stemmed_text)
