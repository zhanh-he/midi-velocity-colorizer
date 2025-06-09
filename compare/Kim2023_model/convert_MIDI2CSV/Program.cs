// Convert a folder (dataset) MIDI files into CSV files
using System;
using System.IO;
using NAudio.Midi;

class MidiToCsvConverter
{
    static void Main(string[] args)
    {
        // Check for command-line arguments
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: dotnet run <inputFolderPath> <outputFolderPath>");
            return;
        }

        string inputFolderPath = args[0];
        string outputFolderPath = args[1];

        // Verify the input folder exists
        if (!Directory.Exists(inputFolderPath))
        {
            Console.WriteLine("The specified input folder does not exist.");
            return;
        }

        // Create the output folder if it doesn't exist
        Directory.CreateDirectory(outputFolderPath);

        // Get all MIDI files in the input folder
        var midiFiles = Directory.GetFiles(inputFolderPath, "*.mid");
        if (midiFiles.Length == 0)
        {
            Console.WriteLine("No MIDI files found in the specified input folder.");
            return;
        }

        // Process each MIDI file
        foreach (var midiFilePath in midiFiles)
        {
            string midiFileName = Path.GetFileNameWithoutExtension(midiFilePath);
            string csvFilePath = Path.Combine(outputFolderPath, midiFileName + ".csv");

            Console.WriteLine($"Processing {midiFileName}.mid...");

            // Load MIDI file
            MidiFile midiFile = new MidiFile(midiFilePath, false); // false for single-track mode

            using (StreamWriter writer = new StreamWriter(csvFilePath))
            {
                // Write CSV header
                writer.WriteLine("time,time_diff,note_num,note_num_diff,low_octave,length,velocity");

                // Variables to store previous values for calculating time_diff and note_num_diff
                int previousTime = 0;
                int previousNote = 0;
                bool isFirstNote = true;

                // Process each track
                for (int track = 0; track < midiFile.Tracks; track++)
                {
                    // Loop through events in each track
                    for (int i = 0; i < midiFile.Events[track].Count; i++)
                    {
                        if (midiFile.Events[track][i] is NoteOnEvent noteOnEvent && noteOnEvent.Velocity > 0)
                        {
                            int time = (int)noteOnEvent.AbsoluteTime;
                            int time_diff = time - previousTime;
                            int note_num = noteOnEvent.NoteNumber;
                            int note_num_diff = isFirstNote ? 0 : note_num - previousNote;
                            int low_octave = note_num < 72 ? 1 : 0; // 1 if note is below 72, otherwise 0
                            int velocity = noteOnEvent.Velocity;
                            int length = 0; // Initialize length

                            // Find the first NoteOffEvent after this NoteOnEvent with the same pitch
                            for (int j = i + 1; j < midiFile.Events[track].Count; j++)
                            {
                                if (midiFile.Events[track][j] is NoteEvent noteOffEvent &&
                                    noteOffEvent.CommandCode == MidiCommandCode.NoteOff &&
                                    noteOffEvent.NoteNumber == note_num)
                                {
                                    // Calculate the length as the time difference between NoteOff and NoteOn
                                    length = (int)(noteOffEvent.AbsoluteTime - noteOnEvent.AbsoluteTime);
                                    break;
                                }
                            }

                            // Write data row to CSV
                            writer.WriteLine($"{time},{time_diff},{note_num},{note_num_diff},{low_octave},{length},{velocity}");

                            // Update previous values
                            previousTime = time;
                            previousNote = note_num;
                            isFirstNote = false; // Set isFirstNote to false after processing the first note
                        }
                    }
                }
            }

            Console.WriteLine($"Converted {midiFileName}.mid to CSV successfully.");
        }

        Console.WriteLine("All MIDI files processed successfully.");
    }
}