# AI Music Generation with MIDI

Generate original music using deep learning! This project uses LSTM neural networks to learn musical patterns from MIDI files and generate new compositions.

---

## 📁 Folder Structure

music-ai-project/
├── midi_songs/ - Place your .mid files here (classical, jazz, etc.)
├── data/ - Automatically generated notes.pkl file
├── model/ - Trained model saved here (music_model.h5)
├── output/ - Generated MIDI files saved here
├── music_generator.py - Main script
└── README.md - You're reading this!

markdown
Copy
Edit

## ⚙️ Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install music21 tensorflow numpy keras
▶️ How to Run
Add MIDI files (.mid) to midi_songs/ folder

Run the script:

bash
Copy
Edit
python music_generator.py
Generated MIDI saved at output/generated_music.mid

🧠 How It Works
Extract notes using music21

Train LSTM on sequences

Predict next notes

Convert predictions to MIDI

🔗 MIDI Sources
https://bitmidi.com

https://magenta.tensorflow.org/datasets/maestro

https://colinraffel.com/projects/lmd/

📦 Sample Download Command
bash
Copy
Edit
curl -o midi_songs/fur_elise.mid https://bitmidi.com/uploads/73533.mid
❗ Notes
Use proper .mid files with real notes

Try 1–2 small files to start

📜 License
MIT License

🙋‍♂️ Credits
Built with:

TensorFlow

music21