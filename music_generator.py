import glob
import numpy as np
import pickle
import random
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation

# Step 1: Extract notes from MIDI files
def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open("data/notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    return notes

# Step 2: Prepare sequences for training
def prepare_sequences(notes, sequence_length=100):
    pitch_names = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(len(pitch_names))
    network_output = np.eye(len(pitch_names))[network_output]

    return network_input, network_output, note_to_int, pitch_names

# Step 3: Create the LSTM model
def create_network(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 4: Train the model
def train_model(model, network_input, network_output):
    model.fit(network_input, network_output, epochs=100, batch_size=64)
    model.save("model/music_model.h5")

# Step 5: Generate new music
def generate_music(model, network_input, note_to_int, pitch_names, output_length=200):
    int_to_note = {num: note for note, num in note_to_int.items()}
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    for _ in range(output_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, [[index / float(len(pitch_names))]], axis=0)
        pattern = pattern[1:]

    return prediction_output

# Step 6: Convert generated notes to MIDI
def create_midi(prediction_output, filename="output/generated_music.mid"):
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            output_notes.append(new_note)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

# Run all steps
if __name__ == "__main__":
    notes = get_notes()
    if len(notes) < 100:
        print("❌ Not enough notes found. Please add more MIDI files to 'midi_songs/' folder.")
        exit()

    network_input, network_output, note_to_int, pitch_names = prepare_sequences(notes)
    if len(network_input) == 0:
        print("❌ No valid sequences generated. Make sure MIDI files contain enough notes.")
        exit()

    model = create_network((network_input.shape[1], network_input.shape[2]), len(pitch_names))
    train_model(model, network_input, network_output)
    prediction_output = generate_music(model, network_input, note_to_int, pitch_names)
    create_midi(prediction_output)
