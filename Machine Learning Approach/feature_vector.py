class feature_vector:

    def __init__(self, season_id="-1", episode_id="-1", speaker_id="-1", word_vector="[-1]",e_id="-1"):

        self.season_id = season_id
        self.episode_id = episode_id
        self.speaker_id = speaker_id
        self.word_vector = word_vector
        self.e_id = e_id

    def get_vector_representation(self):
        vector = [self.season_id, self.episode_id, self.speaker_id] + self.word_vector.tolist() + [self.e_id]
        return vector

