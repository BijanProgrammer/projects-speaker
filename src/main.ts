import fs from 'fs/promises';

import {pipeline} from '@xenova/transformers';
import type {AudioPipelineInputs} from '@xenova/transformers/types/pipelines';

import wavefile from 'wavefile';

async function generateAudioData(): Promise<AudioPipelineInputs> {
    const buffer = await fs.readFile('ignore/audios/jfk.wav');

    const wav = new wavefile.WaveFile(buffer);
    wav.toBitDepth('32f'); // Pipeline expects input as a Float32Array
    wav.toSampleRate(16000); // Whisper expects audio with a sampling rate of 16000

    let audioData = wav.getSamples();

    if (!Array.isArray(audioData)) {
        return audioData;
    }

    if (audioData.length > 1) {
        const SCALING_FACTOR = Math.sqrt(2);

        // Merge channels (into first channel to save memory)
        for (let i = 0; i < audioData[0].length; ++i) {
            audioData[0][i] = (SCALING_FACTOR * (audioData[0][i] + audioData[1][i])) / 2;
        }
    }

    // Select first channel
    audioData = audioData[0];

    return audioData;
}

async function execute(transcriber, audioData): Promise<void> {
    const start = performance.now();
    const output = await transcriber._call(audioData);
    const end = performance.now();

    console.log(`Execution duration: ${(end - start) / 1000} seconds`);
    console.log(output);
}

async function main(): Promise<void> {
    const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
    const audioData = await generateAudioData();
    await execute(transcriber, audioData);
}

main().then();
