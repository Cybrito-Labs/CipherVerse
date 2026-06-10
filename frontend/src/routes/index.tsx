import { lazy, Suspense } from 'react';
import { createBrowserRouter } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { LoadingState } from '@/components/shared/LoadingState';

// Lazy load pages
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const ClassicalCiphers = lazy(() => import('@/pages/ClassicalCiphers'));
const CaesarPage = lazy(() => import('@/pages/classical/CaesarPage'));
const VigenerePage = lazy(() => import('@/pages/classical/VigenerePage'));
const AtbashPage = lazy(() => import('@/pages/classical/AtbashPage'));
const BaconPage = lazy(() => import('@/pages/classical/BaconPage'));
const BifidPage = lazy(() => import('@/pages/classical/BifidPage'));
const AffinePage = lazy(() => import('@/pages/classical/AffinePage'));
const A1Z26Page = lazy(() => import('@/pages/classical/A1Z26Page'));
const RailFencePage = lazy(() => import('@/pages/classical/RailFencePage'));
const SubstitutionPage = lazy(() => import('@/pages/classical/SubstitutionPage'));
const EncodingPage = lazy(() => import('@/pages/EncodingPage'));
const EncodingToolPage = lazy(() => import('@/pages/encoding/EncodingToolPage'));
const SymmetricPage = lazy(() => import('@/pages/SymmetricPage'));
const AesPage = lazy(() => import('@/pages/symmetric/AesPage'));
const DesPage = lazy(() => import('@/pages/symmetric/DesPage'));
const TripleDesPage = lazy(() => import('@/pages/symmetric/TripleDesPage'));
const Rc2Page = lazy(() => import('@/pages/symmetric/Rc2Page'));
const Rc4Page = lazy(() => import('@/pages/symmetric/Rc4Page'));
const Rc4DropPage = lazy(() => import('@/pages/symmetric/Rc4DropPage'));
const BlowfishPage = lazy(() => import('@/pages/symmetric/BlowfishPage'));
const Sm4Page = lazy(() => import('@/pages/symmetric/Sm4Page'));
const CipherSaber2Page = lazy(() => import('@/pages/symmetric/CipherSaber2Page'));
const XorPage = lazy(() => import('@/pages/symmetric/XorPage'));
const AsymmetricPage = lazy(() => import('@/pages/AsymmetricPage'));
const RsaPage = lazy(() => import('@/pages/RsaPage'));
const DsaPage = lazy(() => import('@/pages/DsaPage'));
const CertificatesPage = lazy(() => import('@/pages/CertificatesPage'));
const X509Page = lazy(() => import('@/pages/certificates/X509Page'));
const TlsPage = lazy(() => import('@/pages/certificates/TlsPage'));
const FingerprintPage = lazy(() => import('@/pages/certificates/FingerprintPage'));
const HashingPage = lazy(() => import('@/pages/HashingPage'));

const BlockchainPage = lazy(() => import('@/pages/BlockchainPage'));
const BitcoinValidationPage = lazy(() => import('@/pages/blockchain/BitcoinValidationPage'));
const EthereumValidationPage = lazy(() => import('@/pages/blockchain/EthereumValidationPage'));
const MerkleTreePage = lazy(() => import('@/pages/blockchain/MerkleTreePage'));
const WifEncoderPage = lazy(() => import('@/pages/blockchain/WifEncoderPage'));

const SteganographyPage = lazy(() => import('@/pages/SteganographyPage'));
const TextStegoPage = lazy(() => import('@/pages/steganography/TextStegoPage'));
const ImageStegoPage = lazy(() => import('@/pages/steganography/ImageStegoPage'));
const AudioStegoPage = lazy(() => import('@/pages/steganography/AudioStegoPage'));

const MalwareAnalysisPage = lazy(() => import('@/pages/MalwareAnalysisPage'));
const HashAnalysisPage = lazy(() => import('@/pages/malware/HashAnalysisPage'));
const TlshComparePage = lazy(() => import('@/pages/malware/TlshComparePage'));
const PeAnalysisPage = lazy(() => import('@/pages/malware/PeAnalysisPage'));

const FileForensicsPage = lazy(() => import('@/pages/FileForensicsPage'));
const FileHashPage = lazy(() => import('@/pages/forensics/FileHashPage'));
const MultiHashPage = lazy(() => import('@/pages/forensics/MultiHashPage'));
const EntropyPage = lazy(() => import('@/pages/forensics/EntropyPage'));
const RandomnessPage = lazy(() => import('@/pages/forensics/RandomnessPage'));

const UtilitiesPage = lazy(() => import('@/pages/UtilitiesPage'));
const PasswordStrengthPage = lazy(() => import('@/pages/utilities/PasswordStrengthPage'));
const JwtSignPage = lazy(() => import('@/pages/utilities/JwtSignPage'));
const SaltGeneratorPage = lazy(() => import('@/pages/utilities/SaltGeneratorPage'));
const FletcherChecksumPage = lazy(() => import('@/pages/utilities/FletcherChecksumPage'));

const HistoricalPage = lazy(() => import('@/pages/HistoricalPage'));
const EnigmaPage = lazy(() => import('@/pages/historic/EnigmaPage'));
const BombePage = lazy(() => import('@/pages/historic/BombePage'));
const TypexPage = lazy(() => import('@/pages/historic/TypexPage'));

const ApiExplorerPage = lazy(() => import('@/pages/ApiExplorerPage'));
const SettingsPage = lazy(() => import('@/pages/SettingsPage'));
const NotFound = lazy(() => import('@/pages/NotFound'));

function SuspenseWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<LoadingState message="Loading page..." />}>
      {children}
    </Suspense>
  );
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <SuspenseWrapper><Dashboard /></SuspenseWrapper>,
      },
      {
        path: 'classical',
        children: [
          { index: true, element: <SuspenseWrapper><ClassicalCiphers /></SuspenseWrapper> },
          { path: 'caesar', element: <SuspenseWrapper><CaesarPage /></SuspenseWrapper> },
          { path: 'vigenere', element: <SuspenseWrapper><VigenerePage /></SuspenseWrapper> },
          { path: 'atbash', element: <SuspenseWrapper><AtbashPage /></SuspenseWrapper> },
          { path: 'bacon', element: <SuspenseWrapper><BaconPage /></SuspenseWrapper> },
          { path: 'bifid', element: <SuspenseWrapper><BifidPage /></SuspenseWrapper> },
          { path: 'affine', element: <SuspenseWrapper><AffinePage /></SuspenseWrapper> },
          { path: 'a1z26', element: <SuspenseWrapper><A1Z26Page /></SuspenseWrapper> },
          { path: 'rail-fence', element: <SuspenseWrapper><RailFencePage /></SuspenseWrapper> },
          { path: 'substitution', element: <SuspenseWrapper><SubstitutionPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'encoding',
        children: [
          { index: true, element: <SuspenseWrapper><EncodingPage /></SuspenseWrapper> },
          { path: ':tool', element: <SuspenseWrapper><EncodingToolPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'symmetric',
        children: [
          { index: true, element: <SuspenseWrapper><SymmetricPage /></SuspenseWrapper> },
          { path: 'aes', element: <SuspenseWrapper><AesPage /></SuspenseWrapper> },
          { path: 'des', element: <SuspenseWrapper><DesPage /></SuspenseWrapper> },
          { path: '3des', element: <SuspenseWrapper><TripleDesPage /></SuspenseWrapper> },
          { path: 'rc2', element: <SuspenseWrapper><Rc2Page /></SuspenseWrapper> },
          { path: 'rc4', element: <SuspenseWrapper><Rc4Page /></SuspenseWrapper> },
          { path: 'rc4-drop', element: <SuspenseWrapper><Rc4DropPage /></SuspenseWrapper> },
          { path: 'blowfish', element: <SuspenseWrapper><BlowfishPage /></SuspenseWrapper> },
          { path: 'sm4', element: <SuspenseWrapper><Sm4Page /></SuspenseWrapper> },
          { path: 'ciphersaber2', element: <SuspenseWrapper><CipherSaber2Page /></SuspenseWrapper> },
          { path: 'xor', element: <SuspenseWrapper><XorPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'asymmetric',
        children: [
          { index: true, element: <SuspenseWrapper><AsymmetricPage /></SuspenseWrapper> },
          { path: 'rsa', element: <SuspenseWrapper><RsaPage /></SuspenseWrapper> },
          { path: 'dsa', element: <SuspenseWrapper><DsaPage /></SuspenseWrapper> },
        ],
      },
      { path: 'hashing', element: <SuspenseWrapper><HashingPage /></SuspenseWrapper> },
      {
        path: 'certificates',
        children: [
          { index: true, element: <SuspenseWrapper><CertificatesPage /></SuspenseWrapper> },
          { path: 'x509', element: <SuspenseWrapper><X509Page /></SuspenseWrapper> },
          { path: 'tls', element: <SuspenseWrapper><TlsPage /></SuspenseWrapper> },
          { path: 'fingerprint', element: <SuspenseWrapper><FingerprintPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'blockchain',
        children: [
          { index: true, element: <SuspenseWrapper><BlockchainPage /></SuspenseWrapper> },
          { path: 'bitcoin', element: <SuspenseWrapper><BitcoinValidationPage /></SuspenseWrapper> },
          { path: 'ethereum', element: <SuspenseWrapper><EthereumValidationPage /></SuspenseWrapper> },
          { path: 'merkle', element: <SuspenseWrapper><MerkleTreePage /></SuspenseWrapper> },
          { path: 'wif', element: <SuspenseWrapper><WifEncoderPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'steganography',
        children: [
          { index: true, element: <SuspenseWrapper><SteganographyPage /></SuspenseWrapper> },
          { path: 'text', element: <SuspenseWrapper><TextStegoPage /></SuspenseWrapper> },
          { path: 'image', element: <SuspenseWrapper><ImageStegoPage /></SuspenseWrapper> },
          { path: 'audio', element: <SuspenseWrapper><AudioStegoPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'malware-analysis',
        children: [
          { index: true, element: <SuspenseWrapper><MalwareAnalysisPage /></SuspenseWrapper> },
          { path: 'hash', element: <SuspenseWrapper><HashAnalysisPage /></SuspenseWrapper> },
          { path: 'tlsh', element: <SuspenseWrapper><TlshComparePage /></SuspenseWrapper> },
          { path: 'pe', element: <SuspenseWrapper><PeAnalysisPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'file-forensics',
        children: [
          { index: true, element: <SuspenseWrapper><FileForensicsPage /></SuspenseWrapper> },
          { path: 'hash', element: <SuspenseWrapper><FileHashPage /></SuspenseWrapper> },
          { path: 'multi-hash', element: <SuspenseWrapper><MultiHashPage /></SuspenseWrapper> },
          { path: 'entropy', element: <SuspenseWrapper><EntropyPage /></SuspenseWrapper> },
          { path: 'randomness', element: <SuspenseWrapper><RandomnessPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'utilities',
        children: [
          { index: true, element: <SuspenseWrapper><UtilitiesPage /></SuspenseWrapper> },
          { path: 'password', element: <SuspenseWrapper><PasswordStrengthPage /></SuspenseWrapper> },
          { path: 'jwt', element: <SuspenseWrapper><JwtSignPage /></SuspenseWrapper> },
          { path: 'salt', element: <SuspenseWrapper><SaltGeneratorPage /></SuspenseWrapper> },
          { path: 'fletcher16', element: <SuspenseWrapper><FletcherChecksumPage /></SuspenseWrapper> },
        ],
      },
      {
        path: 'historical',
        children: [
          { index: true, element: <SuspenseWrapper><HistoricalPage /></SuspenseWrapper> },
          { path: 'enigma', element: <SuspenseWrapper><EnigmaPage /></SuspenseWrapper> },
          { path: 'bombe', element: <SuspenseWrapper><BombePage /></SuspenseWrapper> },
          { path: 'typex', element: <SuspenseWrapper><TypexPage /></SuspenseWrapper> },
        ],
      },
      { path: 'api-explorer', element: <SuspenseWrapper><ApiExplorerPage /></SuspenseWrapper> },
      { path: 'settings', element: <SuspenseWrapper><SettingsPage /></SuspenseWrapper> },
      { path: '*', element: <SuspenseWrapper><NotFound /></SuspenseWrapper> },
    ],
  },
]);
