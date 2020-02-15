# Smart Compose Sentence Completion

This is based on Gmail smart compose. This feature predicts the next part of the sentence, given half done sentence is typed by the user.
This feature helps faster composing of mails.
Smart Compose is trained with [Enron email dataset](https://data.world/brianray/enron-email-dataset).
Enron email dataset has more than 5 lakhs emails.

The model is built with Bidirectional Encoder Decoder LSTM.
Trained with about 35000 emails as sample data.
