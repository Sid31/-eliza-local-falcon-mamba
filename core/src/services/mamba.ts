import { fileURLToPath } from "url";
import path from "path";
import fs from "fs";
import * as ort from 'onnxruntime-node';

interface QueuedMessage {
    context: string;
    temperature: number;
    stop: string[];
    max_tokens: number;
    frequency_penalty: number;
    presence_penalty: number;
    useGrammar: boolean;
    resolve: (value: any | string | PromiseLike<any | string>) => void;
    reject: (reason?: any) => void;
}

class MambaService {
    private static instance: MambaService | null = null;
    private session: ort.InferenceSession | null = null;
    private modelInitialized: boolean = false;
    private messageQueue: QueuedMessage[] = [];
    private isProcessing: boolean = false;

    private constructor() {
        console.log("Constructing Mamba Service");
        this.initializeModel();
    }

    public static getInstance(): MambaService {
        if (!MambaService.instance) {
            MambaService.instance = new MambaService();
        }
        return MambaService.instance;
    }

    async initializeModel() {
        try {
            console.log("Initializing Mamba model");

            // Create an inference session
            const modelPath = path.join(__dirname, 'model.onnx');
            this.session = await ort.InferenceSession.create(modelPath);

            this.modelInitialized = true;
            this.processQueue();
            console.log("Mamba model initialized successfully");
        } catch (error) {
            console.error("Error initializing Mamba model:", error);
            throw error;
        }
    }

    async queueMessageCompletion(
        context: string,
        temperature: number,
        stop: string[],
        frequency_penalty: number,
        presence_penalty: number,
        max_tokens: number
    ): Promise<any> {
        return new Promise((resolve, reject) => {
            this.messageQueue.push({
                context,
                temperature,
                stop,
                frequency_penalty,
                presence_penalty,
                max_tokens,
                useGrammar: true,
                resolve,
                reject,
            });
            this.processQueue();
        });
    }

    async queueTextCompletion(
        context: string,
        temperature: number,
        stop: string[],
        frequency_penalty: number,
        presence_penalty: number,
        max_tokens: number
    ): Promise<string> {
        return new Promise((resolve, reject) => {
            this.messageQueue.push({
                context,
                temperature,
                stop,
                frequency_penalty,
                presence_penalty,
                max_tokens,
                useGrammar: false,
                resolve,
                reject,
            });
            this.processQueue();
        });
    }

    private async processQueue() {
        if (
            this.isProcessing ||
            this.messageQueue.length === 0 ||
            !this.modelInitialized
        ) {
            return;
        }

        this.isProcessing = true;

        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (message) {
                try {
                    const response = await this.getCompletionResponse(
                        message.context,
                        message.temperature,
                        message.stop,
                        message.frequency_penalty,
                        message.presence_penalty,
                        message.max_tokens,
                        message.useGrammar
                    );
                    message.resolve(response);
                } catch (error) {
                    message.reject(error);
                }
            }
        }

        this.isProcessing = false;
    }

    private async getCompletionResponse(
        context: string,
        temperature: number,
        stop: string[],
        frequency_penalty: number,
        presence_penalty: number,
        max_tokens: number,
        useGrammar: boolean
    ): Promise<any | string> {
        if (!this.session) {
            throw new Error("Model not initialized");
        }

        // Convert input to tensor
        const inputTensor = new ort.Tensor(
            'string',
            [context],
            [1]
        );

        // Run inference
        const feeds = { input: inputTensor };
        const outputMap = await this.session.run(feeds);
        const output = outputMap.output.data;

        // Convert output to string
        const response = output[0] as string;

        if (useGrammar) {
            try {
                let jsonString = response.match(/```json(.*?)```/s)?.[1].trim();
                if (!jsonString) {
                    try {
                        jsonString = JSON.stringify(JSON.parse(response));
                    } catch {
                        throw new Error("JSON string not found");
                    }
                }
                const parsedResponse = JSON.parse(jsonString);
                console.log("AI: " + parsedResponse.content);
                return parsedResponse;
            } catch (error) {
                console.error("Error parsing JSON:", error);
                throw error;
            }
        } else {
            console.log("AI: " + response);
            return response;
        }
    }

    async getEmbeddingResponse(input: string): Promise<number[] | undefined> {
        if (!this.session) {
            throw new Error("Model not initialized");
        }

        const inputTensor = new ort.Tensor(
            'string',
            [input],
            [1]
        );

        const feeds = { input: inputTensor };
        const outputMap = await this.session.run(feeds);
        const embeddings = outputMap.embeddings.data as Float32Array;

        return Array.from(embeddings);
    }
}

export default MambaService;