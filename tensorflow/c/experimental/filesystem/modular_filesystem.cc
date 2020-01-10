/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/c/experimental/filesystem/modular_filesystem.h"

#include <algorithm>
#include <string>
#include <utility>

#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/util/ptr_util.h"

// TODO(mihaimaruseac): After all filesystems are converted, all calls to
// methods from `FileSystem` will have to be replaced to calls to private
// methods here, as part of making this class a singleton and the only way to
// register/use filesystems.

namespace tensorflow {

using UniquePtrTo_TF_Status =
    ::std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

Status ModularFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  if (ops_->new_random_access_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support NewRandomAccessFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto file = MakeUnique<TF_RandomAccessFile>();
  std::string translated_name = TranslateName(fname);
  ops_->new_random_access_file(filesystem_.get(), translated_name.c_str(),
                               file.get(), plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularRandomAccessFile>(
        translated_name, std::move(file), random_access_file_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::NewWritableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  if (ops_->new_writable_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support NewWritableFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto file = MakeUnique<TF_WritableFile>();
  std::string translated_name = TranslateName(fname);
  ops_->new_writable_file(filesystem_.get(), translated_name.c_str(),
                          file.get(), plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularWritableFile>(translated_name, std::move(file),
                                              writable_file_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::NewAppendableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  if (ops_->new_appendable_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support NewAppendableFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto file = MakeUnique<TF_WritableFile>();
  std::string translated_name = TranslateName(fname);
  ops_->new_appendable_file(filesystem_.get(), translated_name.c_str(),
                            file.get(), plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularWritableFile>(translated_name, std::move(file),
                                              writable_file_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  if (ops_->new_read_only_memory_region_from_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname,
        " does not support NewReadOnlyMemoryRegionFromFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto region = MakeUnique<TF_ReadOnlyMemoryRegion>();
  std::string translated_name = TranslateName(fname);
  ops_->new_read_only_memory_region_from_file(
      filesystem_.get(), translated_name.c_str(), region.get(),
      plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularReadOnlyMemoryRegion>(
        std::move(region), read_only_memory_region_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::FileExists(const std::string& fname) {
  if (ops_->path_exists == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support FileExists()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  const std::string translated_name = TranslateName(fname);
  ops_->path_exists(filesystem_.get(), translated_name.c_str(),
                    plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

bool ModularFileSystem::FilesExist(const std::vector<std::string>& files,
                                   std::vector<Status>* status) {
  if (ops_->paths_exist == nullptr)
    return FileSystem::FilesExist(files, status);

  std::vector<char*> translated_names;
  translated_names.reserve(files.size());
  for (int i = 0; i < files.size(); i++)
    translated_names.push_back(strdup(TranslateName(files[i]).c_str()));

  bool result;
  if (status == nullptr) {
    result = ops_->paths_exist(filesystem_.get(), translated_names.data(),
                               files.size(), nullptr);
  } else {
    std::vector<TF_Status*> plugin_status;
    plugin_status.reserve(files.size());
    for (int i = 0; i < files.size(); i++)
      plugin_status.push_back(TF_NewStatus());
    result = ops_->paths_exist(filesystem_.get(), translated_names.data(),
                               files.size(), plugin_status.data());
    for (int i = 0; i < files.size(); i++) {
      status->push_back(StatusFromTF_Status(plugin_status[i]));
      TF_DeleteStatus(plugin_status[i]);
    }
  }

  for (int i = 0; i < files.size(); i++) free(translated_names[i]);

  return result;
}

Status ModularFileSystem::GetChildren(const std::string& dir,
                                      std::vector<std::string>* result) {
  if (ops_->get_children == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", dir, " does not support GetChildren()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dir);
  char** children;
  const int num_children =
      ops_->get_children(filesystem_.get(), translated_name.c_str(), &children,
                         plugin_status.get());
  if (num_children >= 0) {
    for (int i = 0; i < num_children; i++) {
      result->push_back(std::string(children[i]));
      free(children[i]);
    }
    free(children);
  }

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::GetMatchingPaths(const std::string& pattern,
                                           std::vector<std::string>* result) {
  if (ops_->get_matching_paths == nullptr)
    return internal::GetMatchingPaths(this, Env::Default(), pattern, result);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  char** matches;
  const int num_matches = ops_->get_matching_paths(
      filesystem_.get(), pattern.c_str(), &matches, plugin_status.get());
  if (num_matches >= 0) {
    for (int i = 0; i < num_matches; i++) {
      result->push_back(std::string(matches[i]));
      free(matches[i]);
    }
    free(matches);
  }

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::DeleteFile(const std::string& fname) {
  if (ops_->delete_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support DeleteFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(fname);
  ops_->delete_file(filesystem_.get(), translated_name.c_str(),
                    plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::DeleteRecursively(const std::string& dirname,
                                            int64* undeleted_files,
                                            int64* undeleted_dirs) {
  if (undeleted_files == nullptr || undeleted_dirs == nullptr)
    return errors::FailedPrecondition(
        "DeleteRecursively must not be called with `undeleted_files` or "
        "`undeleted_dirs` set to NULL");

  if (ops_->delete_recursively == nullptr)
    return FileSystem::DeleteRecursively(dirname, undeleted_files,
                                         undeleted_dirs);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  uint64_t plugin_undeleted_files, plugin_undeleted_dirs;
  ops_->delete_recursively(filesystem_.get(), translated_name.c_str(),
                           &plugin_undeleted_files, &plugin_undeleted_dirs,
                           plugin_status.get());
  *undeleted_files = plugin_undeleted_files;
  *undeleted_dirs = plugin_undeleted_dirs;
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::DeleteDir(const std::string& dirname) {
  if (ops_->delete_dir == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", dirname, " does not support DeleteDir()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  ops_->delete_dir(filesystem_.get(), translated_name.c_str(),
                   plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::RecursivelyCreateDir(const std::string& dirname) {
  if (ops_->recursively_create_dir == nullptr)
    return FileSystem::RecursivelyCreateDir(dirname);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  ops_->recursively_create_dir(filesystem_.get(), translated_name.c_str(),
                               plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::CreateDir(const std::string& dirname) {
  if (ops_->create_dir == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", dirname, " does not support CreateDir()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  ops_->create_dir(filesystem_.get(), translated_name.c_str(),
                   plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::Stat(const std::string& fname, FileStatistics* stat) {
  if (ops_->stat == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support Stat()"));

  if (stat == nullptr)
    return errors::InvalidArgument("FileStatistics pointer must not be NULL");

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(fname);
  TF_FileStatistics stats;
  ops_->stat(filesystem_.get(), translated_name.c_str(), &stats,
             plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK) {
    stat->length = stats.length;
    stat->mtime_nsec = stats.mtime_nsec;
    stat->is_directory = stats.is_directory;
  }

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::IsDirectory(const std::string& name) {
  if (ops_->is_directory == nullptr) return FileSystem::IsDirectory(name);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(name);
  ops_->is_directory(filesystem_.get(), translated_name.c_str(),
                     plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::GetFileSize(const std::string& fname,
                                      uint64* file_size) {
  if (ops_->get_file_size == nullptr) {
    FileStatistics stat;
    Status status = Stat(fname, &stat);
    if (!status.ok()) return status;
    if (stat.is_directory)
      return errors::FailedPrecondition("Called GetFileSize on a directory");

    *file_size = stat.length;
    return status;
  }

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(fname);
  *file_size = ops_->get_file_size(filesystem_.get(), translated_name.c_str(),
                                   plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::RenameFile(const std::string& src,
                                     const std::string& target) {
  if (ops_->rename_file == nullptr) {
    Status status = CopyFile(src, target);
    if (status.ok()) status = DeleteFile(src);
    return status;
  }

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_src = TranslateName(src);
  std::string translated_target = TranslateName(target);
  ops_->rename_file(filesystem_.get(), translated_src.c_str(),
                    translated_target.c_str(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::CopyFile(const std::string& src,
                                   const std::string& target) {
  if (ops_->copy_file == nullptr) return FileSystem::CopyFile(src, target);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_src = TranslateName(src);
  std::string translated_target = TranslateName(target);
  ops_->copy_file(filesystem_.get(), translated_src.c_str(),
                  translated_target.c_str(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

std::string ModularFileSystem::TranslateName(const std::string& name) const {
  if (ops_->translate_name == nullptr) return FileSystem::TranslateName(name);

  char* p = ops_->translate_name(filesystem_.get(), name.c_str());
  CHECK(p != nullptr) << "TranslateName(" << name << ") returned nullptr";

  std::string ret(p);
  free(p);
  return ret;
}

void ModularFileSystem::FlushCaches() {
  if (ops_->flush_caches != nullptr) ops_->flush_caches(filesystem_.get());
}

Status ModularRandomAccessFile::Read(uint64 offset, size_t n,
                                     StringPiece* result, char* scratch) const {
  if (ops_->read == nullptr)
    return errors::Unimplemented(
        tensorflow::strings::StrCat("Read() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  int64_t read =
      ops_->read(file_.get(), offset, n, scratch, plugin_status.get());
  if (read > 0) *result = StringPiece(scratch, read);
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularRandomAccessFile::Name(StringPiece* result) const {
  *result = filename_;
  return Status::OK();
}

Status ModularWritableFile::Append(StringPiece data) {
  if (ops_->append == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Append() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->append(file_.get(), data.data(), data.size(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Close() {
  if (ops_->close == nullptr)
    return errors::Unimplemented(
        tensorflow::strings::StrCat("Close() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->close(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Flush() {
  if (ops_->flush == nullptr) return Status::OK();

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->flush(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Sync() {
  if (ops_->sync == nullptr) return Flush();

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->sync(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Name(StringPiece* result) const {
  *result = filename_;
  return Status::OK();
}

Status ModularWritableFile::Tell(int64* position) {
  if (ops_->tell == nullptr)
    return errors::Unimplemented(
        tensorflow::strings::StrCat("Tell() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  *position = ops_->tell(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

// ----------------------------------------------------------------------------
// modular filesystem registration and related helpers
// ----------------------------------------------------------------------------

namespace {

// Checks that all schemes provided by a plugin are valid.
static Status ValidateSchemes(const TF_FilesystemPluginInfo *info) {
  for (int i = 0; i < info->num_schemes; i++)
    if (info->schemes[i] == nullptr)
      return errors::InvalidArgument("Attempted to register filesystem with `nullptr` URI scheme");
  return Status::OK();
}

// Checks if the plugin and core ABI numbers match.
//
// If the numbers don't match, plugin cannot be loaded.
static Status CheckABI(int pluginABI, int coreABI, StringPiece where) {
  if (pluginABI != coreABI)
    return errors::FailedPrecondition(strings::StrCat("Plugin ABI (", pluginABI, ") for ", where, " operations doesn't match expected core ABI (", coreABI, "). Plugin cannot be loaded."));
  return Status::OK();
}

// Checks if the plugin and core ABI numbers match, for all operations.
//
// If the numbers don't match, plugin cannot be loaded.
//
// Uses the simpler `CheckABI(int, int, StringPiece)`.
static Status ValidateABI(const TF_FilesystemPluginInfo *info) {
  TF_RETURN_IF_ERROR(CheckABI(info->filesystem_ops_abi, TF_FILESYSTEM_OPS_ABI, "filesystem"));

  if (info->random_access_file_ops != nullptr)
    TF_RETURN_IF_ERROR(CheckABI(info->random_access_file_ops_abi, TF_RANDOM_ACCESS_FILE_OPS_ABI, "random access file"));

  if (info->writable_file_ops != nullptr)
    TF_RETURN_IF_ERROR(CheckABI(info->writable_file_ops_abi, TF_WRITABLE_FILE_OPS_ABI, "writable file"));

  if (info->read_only_memory_region_ops != nullptr)
    TF_RETURN_IF_ERROR(CheckABI(info->read_only_memory_region_ops_abi, TF_READ_ONLY_MEMORY_REGION_OPS_ABI, "read only memory region"));

  return Status::OK();
}

// Checks if the plugin and core API numbers match, logging mismatches.
static void CheckAPI(int plugin_API, int core_API, StringPiece where) {
  if (plugin_API != core_API) {
    VLOG(0) << "Plugin API (" << plugin_API << ") for " << where
            << " operations doesn't match expected core API (" << core_API
            << "). Plugin will be loaded but functionality might be missing.";
  }
}

// Checks if the plugin and core API numbers match, for all operations.
//
// Uses the simpler `CheckAPIHelper(int, int, StringPiece)`.
static void ValidateAPI(const TF_FilesystemPluginInfo *info) {
  CheckAPI(info->filesystem_ops_api, TF_FILESYSTEM_OPS_API, "filesystem");

  if (info->random_access_file_ops != nullptr)
    CheckAPI(info->random_access_file_ops_api, TF_RANDOM_ACCESS_FILE_OPS_API, "random access file");

  if (info->writable_file_ops != nullptr)
    CheckAPI(info->writable_file_ops_api, TF_WRITABLE_FILE_OPS_API, "writable file");

  if (info->read_only_memory_region_ops != nullptr)
    CheckAPI(info->read_only_memory_region_ops_api, TF_READ_ONLY_MEMORY_REGION_OPS_API, "read only memory region");
}

// Validates the filesystem operations supplied by the plugin.
static Status ValidateHelper(const TF_FilesystemOps* ops) {
  if (ops == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without operations");

  if (ops->init == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without `init` operation");

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without `cleanup` operation");

  return Status::OK();
}

// Validates the random access file operations supplied by the plugin.
static Status ValidateHelper(const TF_RandomAccessFileOps* ops) {
  if (ops == nullptr) {
    // We allow filesystems where files can only be written to (from TF code)
    return Status::OK();
  }

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without `cleanup` operation on random access files");

  return Status::OK();
}

// Validates the writable file operations supplied by the plugin.
static Status ValidateHelper(const TF_WritableFileOps* ops) {
  if (ops == nullptr) {
    // We allow read-only filesystems
    return Status::OK();
  }

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without `cleanup` operation on writable files");

  return Status::OK();
}

// Validates the read only memory region operations given by the plugin.
static Status ValidateHelper(const TF_ReadOnlyMemoryRegionOps* ops) {
  if (ops == nullptr) {
    // read only memory region support is always optional
    return Status::OK();
  }

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without `cleanup` operation on read only memory regions");

  if (ops->data == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without `data` operation on read only memory regions");

  if (ops->length == nullptr)
    return errors::FailedPrecondition("Trying to register filesystem without `length` operation on read only memory regions");

  return Status::OK();
}

// Validates the operations supplied by the plugin.
//
// Uses the 4 simpler `ValidateHelper(const TF_...*)` to validate each
// individual function table and then checks that the function table for a
// specific file type exists if the plugin offers support for creating that
// type of files.
static Status ValidateOperations(const TF_FilesystemPluginInfo *info) {
  TF_RETURN_IF_ERROR(ValidateHelper(info->filesystem_ops));
  TF_RETURN_IF_ERROR(ValidateHelper(info->random_access_file_ops));
  TF_RETURN_IF_ERROR(ValidateHelper(info->writable_file_ops));
  TF_RETURN_IF_ERROR(ValidateHelper(info->read_only_memory_region_ops));

  if (info->filesystem_ops->new_random_access_file != nullptr &&
      info->random_access_file_ops == nullptr)
    return errors::FailedPrecondition(
                 "Filesystem allows creation of random access files but no "
                 "operations on them have been supplied.");

  if ((info->filesystem_ops->new_writable_file != nullptr ||
       info->filesystem_ops->new_appendable_file != nullptr) &&
      info->writable_file_ops == nullptr)
    return errors::FailedPrecondition(
                 "Filesystem allows creation of writable files but no "
                 "operations on them have been supplied.");

  if (info->filesystem_ops->new_read_only_memory_region_from_file != nullptr &&
      info->read_only_memory_region_ops == nullptr)
    return errors::FailedPrecondition(
                 "Filesystem allows creation of readonly memory regions but no "
                 "operations on them have been supplied.");

  return Status::OK();
}

// Copies a function table from plugin memory space to core memory space.
//
// This has three benefits:
//   * allows having newer plugins than the current core TensorFlow: the
//     additional entries in the plugin's table are just discarded;
//   * allows having older plugins than the current core TensorFlow (though
//     we are still warning users): the entries that core TensorFlow expects
//     but plugins didn't provide will be set to `nullptr` values and core
//     TensorFlow will know to not call these on behalf of users;
//   * increased security as plugins will not be able to alter function table
//     after loading up. Thus, malicious plugins can't alter functionality to
//     probe for gadgets inside core TensorFlow. We can even protect the area
//     of memory where the copies reside to not allow any more writes to it
//     after all copies are created.
template <typename T>
static std::unique_ptr<const T> CopyToCore(const T* plugin_ops,
                                           size_t plugin_size) {
  if (plugin_ops == nullptr) return nullptr;

  size_t copy_size = sizeof(T);
  if (plugin_size < copy_size) {
    copy_size = plugin_size;
  }

  auto core_ops = tensorflow::MakeUnique<T>();
  memcpy(const_cast<T*>(core_ops.get()), plugin_ops, copy_size);
  return core_ops;
}

// Registers one filesystem from the plugin.
static Status RegisterFilesystem(const char *scheme, const TF_FilesystemPluginInfo *info) {
  // Step 1: Copy all the function tables to core TensorFlow memory space
  auto core_filesystem_ops = CopyToCore<TF_FilesystemOps>(info->filesystem_ops, info->filesystem_ops_size);
  auto core_random_access_file_ops = CopyToCore<TF_RandomAccessFileOps>(info->random_access_file_ops, info->random_access_file_ops_size);
  auto core_writable_file_ops = CopyToCore<TF_WritableFileOps>(info->writable_file_ops, info->writable_file_ops_size);
  auto core_read_only_memory_region_ops = CopyToCore<TF_ReadOnlyMemoryRegionOps>(info->read_only_memory_region_ops, info->read_only_memory_region_ops_size);

  // Step 2: Initialize the opaque filesystem structure
  auto filesystem = tensorflow::MakeUnique<TF_Filesystem>();
  TF_Status* c_status = TF_NewStatus();
  Status status = Status::OK();
  core_filesystem_ops->init(filesystem.get(), c_status);
  status = Status(c_status->status);
  TF_DeleteStatus(c_status);
  if (!status.ok()) return status;

  // Step 3: Actual registration
  return Env::Default()->RegisterFileSystem(
      scheme, tensorflow::MakeUnique<tensorflow::ModularFileSystem>(
                  std::move(filesystem), std::move(core_filesystem_ops),
                  std::move(core_random_access_file_ops),
                  std::move(core_writable_file_ops),
                  std::move(core_read_only_memory_region_ops)));
}

// Registers all filesystems, if plugin is providing valid information.
//
// Extracted to a separate function so that pointers inside `info` are freed
// by the caller regardless of whether validation/registration failed or not.
static Status ValidateAndRegisterFilesystems(const TF_FilesystemPluginInfo *info) {
  // Step 1: Validate plugin supplied data
  TF_RETURN_IF_ERROR(ValidateSchemes(info));
  TF_RETURN_IF_ERROR(ValidateABI(info));
  ValidateAPI(info);  // we just warn on API number mismatch
  TF_RETURN_IF_ERROR(ValidateOperations(info));

  // Step 2: Initialize the filesystem, for every scheme
  for (int i = 0; i < info->num_schemes; i++)
    TF_RETURN_IF_ERROR(RegisterFilesystem(info->schemes[i], info));

  return Status::OK();
}

}  // namespace

Status RegisterFilesystemPlugin(const std::string& dso_path) {
  // Step 1: Load plugin
  Env *env = Env::Default();
  void* dso_handle;
  TF_RETURN_IF_ERROR(env->LoadLibrary(dso_path.c_str(), &dso_handle));

  // Step 2: Load symbol for `TF_InitPlugin`
  void* dso_symbol;
  TF_RETURN_IF_ERROR(env->GetSymbolFromLibrary(dso_handle, "TF_InitPlugin", &dso_symbol));

  // Step 3: Call `TF_InitPlugin`
  TF_FilesystemPluginInfo info;
  memset(&info, 0, sizeof(info));
  (reinterpret_cast<void (*)(TF_FilesystemPluginInfo*)>(dso_symbol))(&info);

  // Step 4: Validate and register all filesystems
  Status status = ValidateAndRegisterFilesystems(&info);

  // Step 5: Cleanup memory
  if(info.filesystem_ops != nullptr) free(info.filesystem_ops);
  if(info.random_access_file_ops != nullptr) free(info.random_access_file_ops);
  if(info.writable_file_ops != nullptr) free(info.writable_file_ops);
  if(info.read_only_memory_region_ops != nullptr) free(info.read_only_memory_region_ops);
  for (int i = 0; i < info.num_schemes; i++)
    free(info.schemes[i]);
  if(info.schemes != nullptr) free(info.schemes);

  // We're done, return status of `ValidateAndRegisterFilesystems`.
  return status;
}

}  // namespace tensorflow
