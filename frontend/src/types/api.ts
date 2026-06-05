// ============ Classical Ciphers ============
export interface CaesarRequest {
  text: string;
  shift: number;
}

export interface VigenereRequest {
  text: string;
  keyword: string;
}

export interface AtbashRequest {
  text: string;
}

export interface BaconRequest {
  text: string;
}

export interface BifidRequest {
  text: string;
  keyword: string;
}

export interface AffineRequest {
  text: string;
  a: number;
  b: number;
}

export interface A1Z26Request {
  text: string;
}

export interface RailFenceRequest {
  text: string;
  rails: number;
}

export interface SubstituteRequest {
  text: string;
  keyword: string;
}

export interface ClassicalResponse {
  result: string;
}

// ============ Symmetric Ciphers ============
export interface TextPasswordModeRequest {
  text: string;
  password: string;
  mode?: string;
}

export interface CiphertextPasswordModeRequest {
  ciphertext: string;
  password: string;
  mode?: string;
}

export interface AESRequest {
  text: string;
  password: string;
  mode: string;
  key_size: number;
}

export interface AESDecryptRequest {
  ciphertext: string;
  password: string;
  mode: string;
  key_size: number;
}

export interface XORRequest {
  text: string;
  key: string;
}

export interface XORDecryptRequest {
  hex_data: string;
  key: string;
}

export interface XORBruteforceRequest {
  hex_data: string;
}

export interface RC4DropRequest {
  text: string;
  key: string;
  drop_n: number;
}

export interface SymmetricResponse {
  result: string;
}

export interface XORBruteforceResponse {
  results: [number, string][];
}

// ============ Hashing ============
export interface HashRequest {
  text: string;
  algorithm: string;
}

export interface HMACRequest {
  message: string;
  key: string;
  algorithm: string;
}

export interface PBKDF2Request {
  password: string;
  iterations: number;
  dklen: number;
  hash_name: string;
}

export interface ScryptRequest {
  password: string;
  n: number;
  r: number;
  p: number;
  dklen: number;
}

export interface HashResponse {
  result: string;
}

export interface KDFResponse {
  salt: string;
  derived_key?: string;
  hash?: string;
  iterations?: number;
  n?: number;
  r?: number;
  p?: number;
}

// ============ Encoding ============
export interface EncodingRequest {
  data: string;
}

export interface EncodingResponse {
  result: string;
}

// ============ Asymmetric ============
export interface RSAKeyResponse {
  public_key: string;
  private_key: string;
}

export interface RSAEncryptRequest {
  text: string;
  public_key: string;
}

export interface RSADecryptRequest {
  encrypted_base64: string;
  private_key: string;
}

export interface RSASignRequest {
  message: string;
  private_key: string;
}

export interface RSAVerifyRequest {
  message: string;
  signature: string;
  public_key: string;
}

export interface AsymmetricResponse {
  result: string;
  valid?: boolean;
}

// ============ Certificates ============
export interface TLSRequest {
  hostname: string;
  port: number;
}

export interface FingerprintRequest {
  data: string;
  algorithm: string;
}

export interface X509Response {
  Subject: string;
  Issuer: string;
  "Serial Number": number;
  "Not Before": string;
  "Not After": string;
  "Fingerprint SHA256": string;
  Version: string;
  Extensions: string[];
}

// ============ File Tools ============
export interface FileHashRequest {
  filepath: string;
  algorithm: string;
}

export interface FileHashResponse {
  hash: string;
}

export interface FileMultiHashResponse {
  hashes: Record<string, string>;
}

export interface EntropyResponse {
  entropy: number;
  interpretation: string;
}

export interface RandomnessResponse {
  entropy: number;
  bit_balance: Record<string, number>;
  runs: number;
  chi_square: number;
}

// ============ Malware Analysis ============
export interface HashAnalysisRequest {
  hash_value: string;
}

export interface SectionInfo {
  Name: string;
  Entropy: number;
  RawSize: number;
}

export interface PEAnalysisResponse {
  MD5: string;
  Imphash: string;
  Sections: SectionInfo[];
}

// ============ Blockchain ============
export interface AddressRequest {
  address: string;
}

export interface AddressResponse {
  Valid: boolean;
  Type?: string;
  Network?: string;
}

export interface MerkleRequest {
  items: string[];
  algorithm: string;
}

export interface MerkleResponse {
  Root: string;
  Levels: string[][];
}

export interface WIFRequest {
  private_key_hex: string;
  compressed: boolean;
  testnet: boolean;
}

// ============ Steganography ============
export interface StegoTextRequest {
  cover_text: string;
  secret: string;
}

export interface StegoImageRequest {
  input_img: string;
  output_img: string;
  secret: string;
}

export interface StegoAudioRequest {
  input_wav: string;
  output_wav: string;
  secret: string;
}

export interface StegoResponse {
  result: string;
}

// ============ Utilities ============
export interface PasswordStrengthRequest {
  password: string;
}

export interface PasswordStrengthResponse {
  entropy: number;
  strength: string;
}

export interface SaltRequest {
  length: number;
}

export interface JWTRequest {
  payload: Record<string, unknown>;
  secret: string;
  algo: string;
  exp_seconds: number;
}

export interface ChecksumRequest {
  data: string;
}

// ============ Historical Machines ============
export interface EnigmaRequest {
  text: string;
  rotor_order: [string, string, string];
  rotor_positions: [number, number, number];
}

export interface BombeRequest {
  ciphertext: string;
  crib: string;
  rotor_order: [string, string, string];
}

export interface TypexRequest {
  text: string;
  rotors: number;
  positions: number[];
}

export interface HistoricResponse {
  result: string;
  matches: string[];
}
