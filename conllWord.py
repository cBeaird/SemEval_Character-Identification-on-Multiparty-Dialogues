class conllWord:

    def __init__(self, doc_id, scene_id, token_id, word, pos, con_tag, lemma,frameset_id, ws, speaker, named_entity,entity_id):
        self.doc_id = doc_id
        self.scene_id = scene_id
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.con_tag = con_tag
        self.lemma = lemma
        self.frameset_id = frameset_id
        self.ws = ws
        self.speaker = speaker
        self.named_entity = named_entity
        self.entity_id = entity_id

    def __str__(self):
        return ("Document ID : " + self.doc_id + "\n" +
                "Scene ID : " + self.scene_id + "\n" +
                "Token ID : " + self.token_id + "\n" +
                "Word : " + self.word + "\n" +
                "Part of Speech: " + self.pos + "\n" +
                "Connel Tag : " + self.con_tag + "\n" +
                "Lemma : " + self.lemma + "\n" +
                "Frameset ID : " + self.frameset_id + "\n" +
                "WS : " + self.ws + "\n" +
                "Speaker : " + self.speaker + "\n" +
                "Named Entity : " + self.named_entity + "\n" +
                "Entity ID : " + self.entity_id + "\n")
