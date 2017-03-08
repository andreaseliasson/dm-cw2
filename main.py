from pre_process import *
from bag_of_words import *
from helpers import tokens_to_string

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

        tokenized_docs = []
        stemmed_docs = []

        for doc in raw_text_list[:2]:  # Restrict to just the first two docs for testing purposes
            tokenized_doc = tokenize_remove_punct(doc)
            filter_tokenized_doc = filter_words(tokenized_doc)
            filter_tokenized_doc_non_short = remove_short_words(filter_tokenized_doc)
            tokenized_docs.append(filter_tokenized_doc_non_short)

            # Stemming
            stemmed_doc = stem_porter(filter_tokenized_doc_non_short)
            stemmed_docs.append(stemmed_doc)
        print(stemmed_docs)

        docs_as_strings = tokens_to_string(stemmed_docs)
        bag_of_words = create_bag_of_words(docs_as_strings)
