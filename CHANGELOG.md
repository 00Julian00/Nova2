## [1.0.0] - 2025-03-20
### Changes
- Initial release

## [1.0.1] - 2025-05-03
### Changes
#### Features
- Added testing script
- You can now set the voice similarity threshold for speaker detection in the transcriptor conditioning object

#### Bug fixes
- nova.py can now be imported from outside the Nova2 folder

## [1.1.0] - 2025-05-24
### Changes
#### Features
- `get_current_model()` in all inference engines has been replaced with the property `model`. You can still call `get_current_model`, but you will get a deprecation warning, as it will be removed in version 2.0.0.
- Tool calls made by the LLM will now be verified, whether they follow the correct structure laid out out in the metadata file. If not, the LLM will be informed about its incorrect call structure and the tool call will not be executed.

#### Bug fixes
- The elevenlabs inference engine now correctly reads the `similarity_boost` and `use_speaker_boost` parameters from the conditioning object.

#### General changes
- The following classes now use a Singleton pattern: `ContextManager`, `MemoryEmbeddingDatabaseManager`, `VoiceDatabaseManager`, `ToolManager`
- Context data is now no longer being written to the disk for every change. It is instead updated in regular intervals (every 2 minutes), and when the program exits.
- Type hinting now follows the pydantic standards.
- The codebase now only uses absolute import paths, instead of relative ones.
- More tests have been added.

## [2.0.0] - 2025-09-21
### Changes
### Features
- The transcriptor now also has an inference engine system.
- Inference engines are now lazily imported, removing the need to install dependencies for unused inference engines.
- Tools have been reworked and are now much easier to create. Read more in the migration guide.
- The reasoning process of reasoning models can now be filtered out.
- Multiple context files are now supported.
- Added a `stay_alive` function to the API that keeps the program running.
- Added `configure_and_apply` methods for all inference engines to the API.
- Added an experimental recovery manager that can prevent crashes of individual components.

### Bug fixes
- Tool calls that come after a tool call that fails to execute will now still be executed instead of being skipped.

### General changes
- `get_current_model()`has been replaced in all inference engines with the property `model`.
- The TTS inference engines now return an `AudioData` object instead of raw audio data.
- Tools now make use of the dependency injection pattern.
- The `LibraryManager` is now a singleton.
- External code (not written directly for the Nova project) has been moved to the `external` folder.
- External code is no longer covered by tests.
- The inference engine folder has been moved to the top level.
- Secrets are now stored in a `.env` instead of an encrypted database.
- The preferred language for the transcriptor is now set in the conditioning object.
- The voice boost step of the transcription pipeline has been removed.
- Reduced speaker embedding generation frequency to once per sentence.
- The transcriptor has been renamed to `STT` in the API.
- The inference engine to use is now defined in the conditioning object.